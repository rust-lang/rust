//! This module implements expansion of delegation items with early resolved paths.
//! It includes a delegation to a free functions:
//!
//! ```ignore (illustrative)
//! reuse module::name { target_expr_template }
//! ```
//!
//! And delegation to a trait methods:
//!
//! ```ignore (illustrative)
//! reuse <Type as Trait>::name { target_expr_template }
//! ```
//!
//! After expansion for both cases we get:
//!
//! ```ignore (illustrative)
//! fn name(
//!     arg0: InferDelegation(sig_id, Input(0)),
//!     arg1: InferDelegation(sig_id, Input(1)),
//!     ...,
//!     argN: InferDelegation(sig_id, Input(N)),
//! ) -> InferDelegation(sig_id, Output) {
//!     callee_path(target_expr_template(arg0), arg1, ..., argN)
//! }
//! ```
//!
//! Where `callee_path` is a path in delegation item e.g. `<Type as Trait>::name`.
//! `sig_id` is a id of item from which the signature is inherited. It may be a delegation
//! item id (`item_id`) in case of impl trait or path resolution id (`path_id`) otherwise.
//!
//! Since we do not have a proper way to obtain function type information by path resolution
//! in AST, we mark each function parameter type as `InferDelegation` and inherit it in `AstConv`.
//!
//! Similarly generics, predicates and header are set to the "default" values.
//! In case of discrepancy with callee function the `NotSupportedDelegation` error will
//! also be emitted in `AstConv`.

use crate::{ImplTraitPosition, ResolverAstLoweringExt};

use super::{ImplTraitContext, LoweringContext, ParamMode};

use ast::visit::Visitor;
use hir::def::{DefKind, PartialRes, Res};
use hir::{BodyId, HirId};
use rustc_ast as ast;
use rustc_ast::*;
use rustc_errors::ErrorGuaranteed;
use rustc_hir as hir;
use rustc_hir::def_id::DefId;
use rustc_middle::span_bug;
use rustc_middle::ty::ResolverAstLowering;
use rustc_span::{symbol::Ident, Span};
use rustc_target::spec::abi;
use std::iter;

pub(crate) struct DelegationResults<'hir> {
    pub body_id: hir::BodyId,
    pub sig: hir::FnSig<'hir>,
    pub generics: &'hir hir::Generics<'hir>,
}

impl<'hir> LoweringContext<'_, 'hir> {
    pub(crate) fn delegation_has_self(&self, item_id: NodeId, path_id: NodeId, span: Span) -> bool {
        let sig_id = self.get_delegation_sig_id(item_id, path_id, span);
        let Ok(sig_id) = sig_id else {
            return false;
        };
        if let Some(local_sig_id) = sig_id.as_local() {
            self.resolver.has_self.contains(&local_sig_id)
        } else {
            match self.tcx.def_kind(sig_id) {
                DefKind::Fn => false,
                DefKind::AssocFn => self.tcx.associated_item(sig_id).fn_has_self_parameter,
                _ => span_bug!(span, "unexpected DefKind for delegation item"),
            }
        }
    }

    pub(crate) fn lower_delegation(
        &mut self,
        delegation: &Delegation,
        item_id: NodeId,
    ) -> DelegationResults<'hir> {
        let span = delegation.path.segments.last().unwrap().ident.span;
        let sig_id = self.get_delegation_sig_id(item_id, delegation.id, span);
        match sig_id {
            Ok(sig_id) => {
                let decl = self.lower_delegation_decl(sig_id, span);
                let sig = self.lower_delegation_sig(span, decl);
                let body_id = self.lower_delegation_body(sig.decl, delegation);

                let generics = self.lower_delegation_generics(span);
                DelegationResults { body_id, sig, generics }
            }
            Err(err) => self.generate_delegation_error(err, span),
        }
    }

    fn get_delegation_sig_id(
        &self,
        item_id: NodeId,
        path_id: NodeId,
        span: Span,
    ) -> Result<DefId, ErrorGuaranteed> {
        let sig_id = if self.is_in_trait_impl { item_id } else { path_id };
        let sig_id = self
            .resolver
            .get_partial_res(sig_id)
            .map(|r| r.expect_full_res().opt_def_id())
            .unwrap_or(None);

        sig_id.ok_or_else(|| {
            self.tcx
                .dcx()
                .span_delayed_bug(span, "LoweringContext: couldn't resolve delegation item")
        })
    }

    fn lower_delegation_generics(&mut self, span: Span) -> &'hir hir::Generics<'hir> {
        self.arena.alloc(hir::Generics {
            params: &[],
            predicates: &[],
            has_where_clause_predicates: false,
            where_clause_span: span,
            span: span,
        })
    }

    fn lower_delegation_decl(
        &mut self,
        sig_id: DefId,
        param_span: Span,
    ) -> &'hir hir::FnDecl<'hir> {
        let args_count = if let Some(local_sig_id) = sig_id.as_local() {
            // Map may be filled incorrectly due to recursive delegation.
            // Error will be emmited later in astconv.
            self.resolver.fn_parameter_counts.get(&local_sig_id).cloned().unwrap_or_default()
        } else {
            self.tcx.fn_arg_names(sig_id).len()
        };
        let inputs = self.arena.alloc_from_iter((0..args_count).into_iter().map(|arg| hir::Ty {
            hir_id: self.next_id(),
            kind: hir::TyKind::InferDelegation(sig_id, hir::InferDelegationKind::Input(arg)),
            span: self.lower_span(param_span),
        }));

        let output = self.arena.alloc(hir::Ty {
            hir_id: self.next_id(),
            kind: hir::TyKind::InferDelegation(sig_id, hir::InferDelegationKind::Output),
            span: self.lower_span(param_span),
        });

        self.arena.alloc(hir::FnDecl {
            inputs,
            output: hir::FnRetTy::Return(output),
            c_variadic: false,
            lifetime_elision_allowed: true,
            implicit_self: hir::ImplicitSelfKind::None,
        })
    }

    fn lower_delegation_sig(
        &mut self,
        span: Span,
        decl: &'hir hir::FnDecl<'hir>,
    ) -> hir::FnSig<'hir> {
        hir::FnSig {
            decl,
            header: hir::FnHeader {
                unsafety: hir::Unsafety::Normal,
                constness: hir::Constness::NotConst,
                asyncness: hir::IsAsync::NotAsync,
                abi: abi::Abi::Rust,
            },
            span: self.lower_span(span),
        }
    }

    fn generate_param(&mut self, ty: &'hir hir::Ty<'hir>) -> (hir::Param<'hir>, NodeId) {
        let pat_node_id = self.next_node_id();
        let pat_id = self.lower_node_id(pat_node_id);
        let pat = self.arena.alloc(hir::Pat {
            hir_id: pat_id,
            kind: hir::PatKind::Binding(hir::BindingAnnotation::NONE, pat_id, Ident::empty(), None),
            span: ty.span,
            default_binding_modes: false,
        });

        (hir::Param { hir_id: self.next_id(), pat, ty_span: ty.span, span: ty.span }, pat_node_id)
    }

    fn generate_arg(&mut self, ty: &'hir hir::Ty<'hir>, param_id: HirId) -> hir::Expr<'hir> {
        let segments = self.arena.alloc_from_iter(iter::once(hir::PathSegment {
            ident: Ident::empty(),
            hir_id: self.next_id(),
            res: Res::Local(param_id),
            args: None,
            infer_args: false,
        }));

        let path =
            self.arena.alloc(hir::Path { span: ty.span, res: Res::Local(param_id), segments });

        hir::Expr {
            hir_id: self.next_id(),
            kind: hir::ExprKind::Path(hir::QPath::Resolved(None, path)),
            span: ty.span,
        }
    }

    fn lower_delegation_body(
        &mut self,
        decl: &'hir hir::FnDecl<'hir>,
        delegation: &Delegation,
    ) -> BodyId {
        let path = self.lower_qpath(
            delegation.id,
            &delegation.qself,
            &delegation.path,
            ParamMode::Optional,
            ImplTraitContext::Disallowed(ImplTraitPosition::Path),
            None,
        );
        let block = delegation.body.as_deref();

        self.lower_body(|this| {
            let mut parameters: Vec<hir::Param<'_>> = Vec::new();
            let mut args: Vec<hir::Expr<'hir>> = Vec::new();

            for (idx, param_ty) in decl.inputs.iter().enumerate() {
                let (param, pat_node_id) = this.generate_param(param_ty);
                parameters.push(param);

                let arg = if let Some(block) = block
                    && idx == 0
                {
                    let mut self_resolver = SelfResolver {
                        resolver: this.resolver,
                        path_id: delegation.id,
                        self_param_id: pat_node_id,
                    };
                    self_resolver.visit_block(block);
                    let block = this.lower_block(block, false);
                    hir::Expr {
                        hir_id: this.next_id(),
                        kind: hir::ExprKind::Block(block, None),
                        span: block.span,
                    }
                } else {
                    let pat_hir_id = this.lower_node_id(pat_node_id);
                    this.generate_arg(param_ty, pat_hir_id)
                };
                args.push(arg);
            }

            let args = self.arena.alloc_from_iter(args);
            let final_expr = this.generate_call(path, args);
            (this.arena.alloc_from_iter(parameters), final_expr)
        })
    }

    fn generate_call(
        &mut self,
        path: hir::QPath<'hir>,
        args: &'hir [hir::Expr<'hir>],
    ) -> hir::Expr<'hir> {
        let callee = self.arena.alloc(hir::Expr {
            hir_id: self.next_id(),
            kind: hir::ExprKind::Path(path),
            span: path.span(),
        });

        let expr = self.arena.alloc(hir::Expr {
            hir_id: self.next_id(),
            kind: hir::ExprKind::Call(callee, args),
            span: path.span(),
        });

        let block = self.arena.alloc(hir::Block {
            stmts: &[],
            expr: Some(expr),
            hir_id: self.next_id(),
            rules: hir::BlockCheckMode::DefaultBlock,
            span: path.span(),
            targeted_by_break: false,
        });

        hir::Expr {
            hir_id: self.next_id(),
            kind: hir::ExprKind::Block(block, None),
            span: path.span(),
        }
    }

    fn generate_delegation_error(
        &mut self,
        err: ErrorGuaranteed,
        span: Span,
    ) -> DelegationResults<'hir> {
        let generics = self.lower_delegation_generics(span);

        let decl = self.arena.alloc(hir::FnDecl {
            inputs: &[],
            output: hir::FnRetTy::DefaultReturn(span),
            c_variadic: false,
            lifetime_elision_allowed: true,
            implicit_self: hir::ImplicitSelfKind::None,
        });

        let sig = self.lower_delegation_sig(span, decl);
        let body_id = self.lower_body(|this| {
            let expr =
                hir::Expr { hir_id: this.next_id(), kind: hir::ExprKind::Err(err), span: span };
            (&[], expr)
        });
        DelegationResults { generics, body_id, sig }
    }
}

struct SelfResolver<'a> {
    resolver: &'a mut ResolverAstLowering,
    path_id: NodeId,
    self_param_id: NodeId,
}

impl<'a> SelfResolver<'a> {
    fn try_replace_id(&mut self, id: NodeId) {
        if let Some(res) = self.resolver.partial_res_map.get(&id)
            && let Some(Res::Local(sig_id)) = res.full_res()
            && sig_id == self.path_id
        {
            let new_res = PartialRes::new(Res::Local(self.self_param_id));
            self.resolver.partial_res_map.insert(id, new_res);
        }
    }
}

impl<'ast, 'a> Visitor<'ast> for SelfResolver<'a> {
    fn visit_path(&mut self, path: &'ast Path, id: NodeId) {
        self.try_replace_id(id);
        visit::walk_path(self, path);
    }

    fn visit_path_segment(&mut self, seg: &'ast PathSegment) {
        self.try_replace_id(seg.id);
        visit::walk_path_segment(self, seg);
    }
}
