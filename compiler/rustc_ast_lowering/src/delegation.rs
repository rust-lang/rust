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
//! in AST, we mark each function parameter type as `InferDelegation` and inherit it during
//! HIR ty lowering.
//!
//! Similarly generics, predicates and header are set to the "default" values.
//! In case of discrepancy with callee function the `UnsupportedDelegation` error will
//! also be emitted during HIR ty lowering.

use std::iter;

use ast::visit::Visitor;
use hir::def::{DefKind, PartialRes, Res};
use hir::{BodyId, HirId};
use rustc_abi::ExternAbi;
use rustc_ast::*;
use rustc_errors::ErrorGuaranteed;
use rustc_hir::def_id::DefId;
use rustc_middle::span_bug;
use rustc_middle::ty::{Asyncness, ResolverAstLowering};
use rustc_span::{Ident, Span, Symbol};
use {rustc_ast as ast, rustc_hir as hir};

use super::{GenericArgsMode, ImplTraitContext, LoweringContext, ParamMode};
use crate::{AllowReturnTypeNotation, ImplTraitPosition, ResolverAstLoweringExt};

pub(crate) struct DelegationResults<'hir> {
    pub body_id: hir::BodyId,
    pub sig: hir::FnSig<'hir>,
    pub ident: Ident,
    pub generics: &'hir hir::Generics<'hir>,
}

impl<'hir> LoweringContext<'_, 'hir> {
    /// Defines whether the delegatee is an associated function whose first parameter is `self`.
    pub(crate) fn delegatee_is_method(
        &self,
        item_id: NodeId,
        path_id: NodeId,
        span: Span,
        is_in_trait_impl: bool,
    ) -> bool {
        let sig_id = self.get_delegation_sig_id(item_id, path_id, span, is_in_trait_impl);
        let Ok(sig_id) = sig_id else {
            return false;
        };
        self.is_method(sig_id, span)
    }

    fn is_method(&self, def_id: DefId, span: Span) -> bool {
        match self.tcx.def_kind(def_id) {
            DefKind::Fn => false,
            DefKind::AssocFn => match def_id.as_local() {
                Some(local_def_id) => self
                    .resolver
                    .delegation_fn_sigs
                    .get(&local_def_id)
                    .is_some_and(|sig| sig.has_self),
                None => self.tcx.associated_item(def_id).is_method(),
            },
            _ => span_bug!(span, "unexpected DefKind for delegation item"),
        }
    }

    pub(crate) fn lower_delegation(
        &mut self,
        delegation: &Delegation,
        item_id: NodeId,
        is_in_trait_impl: bool,
    ) -> DelegationResults<'hir> {
        let span = self.lower_span(delegation.path.segments.last().unwrap().ident.span);
        let sig_id = self.get_delegation_sig_id(item_id, delegation.id, span, is_in_trait_impl);
        match sig_id {
            Ok(sig_id) => {
                let (param_count, c_variadic) = self.param_count(sig_id);
                let decl = self.lower_delegation_decl(sig_id, param_count, c_variadic, span);
                let sig = self.lower_delegation_sig(sig_id, decl, span);
                let body_id = self.lower_delegation_body(delegation, param_count, span);
                let ident = self.lower_ident(delegation.ident);
                let generics = self.lower_delegation_generics(span);
                DelegationResults { body_id, sig, ident, generics }
            }
            Err(err) => self.generate_delegation_error(err, span),
        }
    }

    fn get_delegation_sig_id(
        &self,
        item_id: NodeId,
        path_id: NodeId,
        span: Span,
        is_in_trait_impl: bool,
    ) -> Result<DefId, ErrorGuaranteed> {
        let sig_id = if is_in_trait_impl { item_id } else { path_id };
        self.get_resolution_id(sig_id, span)
    }

    fn get_resolution_id(&self, node_id: NodeId, span: Span) -> Result<DefId, ErrorGuaranteed> {
        let def_id =
            self.resolver.get_partial_res(node_id).and_then(|r| r.expect_full_res().opt_def_id());
        def_id.ok_or_else(|| {
            self.tcx.dcx().span_delayed_bug(
                span,
                format!("LoweringContext: couldn't resolve node {:?} in delegation item", node_id),
            )
        })
    }

    fn lower_delegation_generics(&mut self, span: Span) -> &'hir hir::Generics<'hir> {
        self.arena.alloc(hir::Generics {
            params: &[],
            predicates: &[],
            has_where_clause_predicates: false,
            where_clause_span: span,
            span,
        })
    }

    // Function parameter count, including C variadic `...` if present.
    fn param_count(&self, sig_id: DefId) -> (usize, bool /*c_variadic*/) {
        if let Some(local_sig_id) = sig_id.as_local() {
            // Map may be filled incorrectly due to recursive delegation.
            // Error will be emitted later during HIR ty lowering.
            match self.resolver.delegation_fn_sigs.get(&local_sig_id) {
                Some(sig) => (sig.param_count, sig.c_variadic),
                None => (0, false),
            }
        } else {
            let sig = self.tcx.fn_sig(sig_id).skip_binder().skip_binder();
            (sig.inputs().len() + usize::from(sig.c_variadic), sig.c_variadic)
        }
    }

    fn lower_delegation_decl(
        &mut self,
        sig_id: DefId,
        param_count: usize,
        c_variadic: bool,
        span: Span,
    ) -> &'hir hir::FnDecl<'hir> {
        // The last parameter in C variadic functions is skipped in the signature,
        // like during regular lowering.
        let decl_param_count = param_count - c_variadic as usize;
        let inputs = self.arena.alloc_from_iter((0..decl_param_count).map(|arg| hir::Ty {
            hir_id: self.next_id(),
            kind: hir::TyKind::InferDelegation(sig_id, hir::InferDelegationKind::Input(arg)),
            span,
        }));

        let output = self.arena.alloc(hir::Ty {
            hir_id: self.next_id(),
            kind: hir::TyKind::InferDelegation(sig_id, hir::InferDelegationKind::Output),
            span,
        });

        self.arena.alloc(hir::FnDecl {
            inputs,
            output: hir::FnRetTy::Return(output),
            c_variadic,
            lifetime_elision_allowed: true,
            implicit_self: hir::ImplicitSelfKind::None,
        })
    }

    fn lower_delegation_sig(
        &mut self,
        sig_id: DefId,
        decl: &'hir hir::FnDecl<'hir>,
        span: Span,
    ) -> hir::FnSig<'hir> {
        let header = if let Some(local_sig_id) = sig_id.as_local() {
            match self.resolver.delegation_fn_sigs.get(&local_sig_id) {
                Some(sig) => {
                    let parent = self.tcx.parent(sig_id);
                    // HACK: we override the default safety instead of generating attributes from the ether.
                    // We are not forwarding the attributes, as the delegation fn sigs are collected on the ast,
                    // and here we need the hir attributes.
                    let default_safety =
                        if sig.target_feature || self.tcx.def_kind(parent) == DefKind::ForeignMod {
                            hir::Safety::Unsafe
                        } else {
                            hir::Safety::Safe
                        };
                    self.lower_fn_header(sig.header, default_safety, &[])
                }
                None => self.generate_header_error(),
            }
        } else {
            let sig = self.tcx.fn_sig(sig_id).skip_binder().skip_binder();
            let asyncness = match self.tcx.asyncness(sig_id) {
                Asyncness::Yes => hir::IsAsync::Async(span),
                Asyncness::No => hir::IsAsync::NotAsync,
            };
            hir::FnHeader {
                safety: if self.tcx.codegen_fn_attrs(sig_id).safe_target_features {
                    hir::HeaderSafety::SafeTargetFeatures
                } else {
                    hir::HeaderSafety::Normal(sig.safety)
                },
                constness: self.tcx.constness(sig_id),
                asyncness,
                abi: sig.abi,
            }
        };
        hir::FnSig { decl, header, span }
    }

    fn generate_param(&mut self, idx: usize, span: Span) -> (hir::Param<'hir>, NodeId) {
        let pat_node_id = self.next_node_id();
        let pat_id = self.lower_node_id(pat_node_id);
        let ident = Ident::with_dummy_span(Symbol::intern(&format!("arg{idx}")));
        let pat = self.arena.alloc(hir::Pat {
            hir_id: pat_id,
            kind: hir::PatKind::Binding(hir::BindingMode::NONE, pat_id, ident, None),
            span,
            default_binding_modes: false,
        });

        (hir::Param { hir_id: self.next_id(), pat, ty_span: span, span }, pat_node_id)
    }

    fn generate_arg(&mut self, idx: usize, param_id: HirId, span: Span) -> hir::Expr<'hir> {
        let segments = self.arena.alloc_from_iter(iter::once(hir::PathSegment {
            ident: Ident::with_dummy_span(Symbol::intern(&format!("arg{idx}"))),
            hir_id: self.next_id(),
            res: Res::Local(param_id),
            args: None,
            infer_args: false,
        }));

        let path = self.arena.alloc(hir::Path { span, res: Res::Local(param_id), segments });
        self.mk_expr(hir::ExprKind::Path(hir::QPath::Resolved(None, path)), span)
    }

    fn lower_delegation_body(
        &mut self,
        delegation: &Delegation,
        param_count: usize,
        span: Span,
    ) -> BodyId {
        let block = delegation.body.as_deref();

        self.lower_body(|this| {
            let mut parameters: Vec<hir::Param<'_>> = Vec::with_capacity(param_count);
            let mut args: Vec<hir::Expr<'_>> = Vec::with_capacity(param_count);

            for idx in 0..param_count {
                let (param, pat_node_id) = this.generate_param(idx, span);
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
                    // Target expr needs to lower `self` path.
                    this.ident_and_label_to_local_id.insert(pat_node_id, param.pat.hir_id.local_id);
                    this.lower_target_expr(&block)
                } else {
                    this.generate_arg(idx, param.pat.hir_id, span)
                };
                args.push(arg);
            }

            let final_expr = this.finalize_body_lowering(delegation, args, span);
            (this.arena.alloc_from_iter(parameters), final_expr)
        })
    }

    // FIXME(fn_delegation): Alternatives for target expression lowering:
    // https://github.com/rust-lang/rfcs/pull/3530#issuecomment-2197170600.
    fn lower_target_expr(&mut self, block: &Block) -> hir::Expr<'hir> {
        if let [stmt] = block.stmts.as_slice()
            && let StmtKind::Expr(expr) = &stmt.kind
        {
            return self.lower_expr_mut(expr);
        }

        let block = self.lower_block(block, false);
        self.mk_expr(hir::ExprKind::Block(block, None), block.span)
    }

    // Generates expression for the resulting body. If possible, `MethodCall` is used
    // to allow autoref/autoderef for target expression. For example in:
    //
    // trait Trait : Sized {
    //     fn by_value(self) -> i32 { 1 }
    //     fn by_mut_ref(&mut self) -> i32 { 2 }
    //     fn by_ref(&self) -> i32 { 3 }
    // }
    //
    // struct NewType(SomeType);
    // impl Trait for NewType {
    //     reuse Trait::* { self.0 }
    // }
    //
    // `self.0` will automatically coerce.
    fn finalize_body_lowering(
        &mut self,
        delegation: &Delegation,
        args: Vec<hir::Expr<'hir>>,
        span: Span,
    ) -> hir::Expr<'hir> {
        let args = self.arena.alloc_from_iter(args);

        let has_generic_args =
            delegation.path.segments.iter().rev().skip(1).any(|segment| segment.args.is_some());

        let call = if self
            .get_resolution_id(delegation.id, span)
            .and_then(|def_id| Ok(self.is_method(def_id, span)))
            .unwrap_or_default()
            && delegation.qself.is_none()
            && !has_generic_args
            && !args.is_empty()
        {
            let ast_segment = delegation.path.segments.last().unwrap();
            let segment = self.lower_path_segment(
                delegation.path.span,
                ast_segment,
                ParamMode::Optional,
                GenericArgsMode::Err,
                ImplTraitContext::Disallowed(ImplTraitPosition::Path),
                None,
            );
            let segment = self.arena.alloc(segment);

            self.arena.alloc(hir::Expr {
                hir_id: self.next_id(),
                kind: hir::ExprKind::MethodCall(segment, &args[0], &args[1..], span),
                span,
            })
        } else {
            let path = self.lower_qpath(
                delegation.id,
                &delegation.qself,
                &delegation.path,
                ParamMode::Optional,
                AllowReturnTypeNotation::No,
                ImplTraitContext::Disallowed(ImplTraitPosition::Path),
                None,
            );

            let callee_path = self.arena.alloc(self.mk_expr(hir::ExprKind::Path(path), span));
            self.arena.alloc(self.mk_expr(hir::ExprKind::Call(callee_path, args), span))
        };
        let block = self.arena.alloc(hir::Block {
            stmts: &[],
            expr: Some(call),
            hir_id: self.next_id(),
            rules: hir::BlockCheckMode::DefaultBlock,
            span,
            targeted_by_break: false,
        });

        self.mk_expr(hir::ExprKind::Block(block, None), span)
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

        let header = self.generate_header_error();
        let sig = hir::FnSig { decl, header, span };

        let ident = Ident::dummy();
        let body_id = self.lower_body(|this| (&[], this.mk_expr(hir::ExprKind::Err(err), span)));
        DelegationResults { ident, generics, body_id, sig }
    }

    fn generate_header_error(&self) -> hir::FnHeader {
        hir::FnHeader {
            safety: hir::Safety::Safe.into(),
            constness: hir::Constness::NotConst,
            asyncness: hir::IsAsync::NotAsync,
            abi: ExternAbi::Rust,
        }
    }

    #[inline]
    fn mk_expr(&mut self, kind: hir::ExprKind<'hir>, span: Span) -> hir::Expr<'hir> {
        hir::Expr { hir_id: self.next_id(), kind, span }
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
    fn visit_id(&mut self, id: NodeId) {
        self.try_replace_id(id);
    }
}
