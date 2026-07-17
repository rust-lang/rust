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
use generics::GenericsGenerationResult;
use hir::HirId;
use hir::def::Res;
use rustc_abi::ExternAbi;
use rustc_ast as ast;
use rustc_ast::*;
use rustc_hir::def::DefKind;
use rustc_hir::{self as hir, FnDeclFlags};
use rustc_middle::ty::Asyncness;
use rustc_span::def_id::DefId;
use rustc_span::symbol::kw;
use rustc_span::{Ident, Span, Symbol, sym};

use crate::delegation::generics::{GenericsGenerationResults, GenericsPosition};
use crate::delegation::resolution::resolver::DelegationResolver;
use crate::delegation::resolution::{DelegationResolution, ParamInfo};
use crate::{
    AllowReturnTypeNotation, ImplTraitContext, ImplTraitPosition, LoweringContext, ParamMode,
};

mod attributes;
mod generics;
mod resolution;

pub(crate) struct DelegationResults<'hir> {
    pub body_id: hir::BodyId,
    pub sig: hir::FnSig<'hir>,
    pub ident: Ident,
    pub generics: &'hir hir::Generics<'hir>,
}

impl<'hir> LoweringContext<'_, 'hir> {
    pub(crate) fn lower_delegation(&mut self, delegation: &Delegation) -> DelegationResults<'hir> {
        let span = self.lower_span(delegation.last_segment_span());

        let resolver = DelegationResolver::new(self);
        let Ok((res, mut generics)) = resolver.resolve_delegation(delegation, span) else {
            return self.generate_delegation_error(span, delegation);
        };

        self.add_attrs_if_needed(&res);

        let (body_id, call_expr_id, unused_target_expr) =
            self.lower_delegation_body(delegation, &res, &mut generics);

        let decl = self.lower_delegation_decl(&res, &generics, call_expr_id, unused_target_expr);

        let sig = self.lower_delegation_sig(res.sig_id, decl, span);

        let ident = self.lower_ident(delegation.ident);

        let generics = self.arena.alloc(hir::Generics {
            has_where_clause_predicates: false,
            params: self.arena.alloc_from_iter(generics.all_params()),
            predicates: self.arena.alloc_from_iter(generics.all_predicates()),
            span,
            where_clause_span: span,
        });

        DelegationResults { body_id, sig, ident, generics }
    }

    fn lower_delegation_decl(
        &mut self,
        res: &DelegationResolution,
        generics: &GenericsGenerationResults<'hir>,
        call_expr_id: HirId,
        unused_target_expr: bool,
    ) -> &'hir hir::FnDecl<'hir> {
        let &DelegationResolution { source, call_path_res, span, sig_id, .. } = res;
        let ParamInfo { param_count, c_variadic, splatted } = res.param_info;

        // The last parameter in C variadic functions is skipped in the signature,
        // like during regular lowering.
        let decl_param_count = param_count - c_variadic as usize;
        let inputs = self.arena.alloc_from_iter((0..decl_param_count).map(|arg| hir::Ty {
            hir_id: self.next_id(),
            kind: hir::TyKind::InferDelegation(hir::InferDelegation::Sig(
                sig_id,
                hir::InferDelegationSig::Input(arg),
            )),
            span,
        }));

        let output = self.arena.alloc(hir::Ty {
            hir_id: self.next_id(),
            kind: hir::TyKind::InferDelegation(hir::InferDelegation::Sig(
                sig_id,
                hir::InferDelegationSig::Output(self.arena.alloc(hir::DelegationInfo {
                    call_expr_id,
                    call_path_res,
                    arguments_to_map: res.sig_mapping.arguments_to_map.clone(),
                    child_seg_id: generics.child.args_segment_id,
                    child_seg_id_for_sig: generics.child.segment_id_for_sig(),
                    parent_seg_id_for_sig: generics.parent.segment_id_for_sig(),
                    self_ty_propagation_kind: generics.self_ty_propagation_kind,
                    group_id: {
                        let id = match source {
                            DelegationSource::Single => None,
                            DelegationSource::List(expn_id) => Some(expn_id),
                            DelegationSource::Glob => {
                                Some(self.tcx.expn_that_defined(self.owner.def_id).expect_local())
                            }
                        };

                        id.map(|id| (id, unused_target_expr))
                    },
                })),
            )),
            span,
        });

        self.arena.alloc(hir::FnDecl {
            inputs,
            output: hir::FnRetTy::Return(output),
            fn_decl_kind: FnDeclFlags::default()
                .set_lifetime_elision_allowed(true)
                .set_c_variadic(c_variadic)
                .set_splatted(splatted, inputs.len())
                .unwrap(),
        })
    }

    fn lower_delegation_sig(
        &mut self,
        sig_id: DefId,
        decl: &'hir hir::FnDecl<'hir>,
        span: Span,
    ) -> hir::FnSig<'hir> {
        let sig = self.tcx.fn_sig(sig_id).skip_binder().skip_binder();
        let asyncness = match self.tcx.asyncness(sig_id) {
            Asyncness::Yes => hir::IsAsync::Async(span),
            Asyncness::No => hir::IsAsync::NotAsync,
        };

        let header = hir::FnHeader {
            safety: if self.tcx.codegen_fn_attrs(sig_id).safe_target_features {
                hir::HeaderSafety::SafeTargetFeatures
            } else {
                hir::HeaderSafety::Normal(sig.safety())
            },
            constness: self.tcx.constness(sig_id),
            asyncness,
            abi: sig.abi(),
        };

        hir::FnSig { decl, header, span }
    }

    fn generate_param(
        &mut self,
        is_method: bool,
        idx: usize,
        span: Span,
    ) -> (hir::Param<'hir>, NodeId) {
        let pat_node_id = self.next_node_id();
        let pat_id = self.lower_node_id(pat_node_id);
        // FIXME(cjgillot) AssocItem currently relies on self parameter being exactly named `self`.
        let name = if is_method && idx == 0 {
            kw::SelfLower
        } else {
            Symbol::intern(&format!("arg{idx}"))
        };
        let ident = Ident::with_dummy_span(name);
        let pat = self.arena.alloc(hir::Pat {
            hir_id: pat_id,
            kind: hir::PatKind::Binding(hir::BindingMode::NONE, pat_id, ident, None),
            span,
            default_binding_modes: false,
        });

        (hir::Param { hir_id: self.next_id(), pat, ty_span: span, span }, pat_node_id)
    }

    fn generate_arg(
        &mut self,
        is_method: bool,
        idx: usize,
        param_id: HirId,
        span: Span,
    ) -> hir::Expr<'hir> {
        // FIXME(cjgillot) AssocItem currently relies on self parameter being exactly named `self`.
        let name = if is_method && idx == 0 {
            kw::SelfLower
        } else {
            Symbol::intern(&format!("arg{idx}"))
        };

        let segments = self.arena.alloc_from_iter(iter::once(hir::PathSegment {
            ident: Ident::with_dummy_span(name),
            hir_id: self.next_id(),
            res: Res::Local(param_id),
            args: None,
            infer_args: false,
            delegation_child_segment: false,
        }));

        let path = self.arena.alloc(hir::Path {
            span,
            res: Res::Local(param_id),
            via_crate: None,
            segments,
        });
        self.mk_expr(hir::ExprKind::Path(hir::QPath::Resolved(None, path)), span)
    }

    fn lower_delegation_body(
        &mut self,
        delegation: &Delegation,
        res: &DelegationResolution,
        generics: &mut GenericsGenerationResults<'hir>,
    ) -> (hir::BodyId, HirId, bool) {
        let block = delegation.body.as_deref();
        let mut call_expr_id = HirId::INVALID;
        let mut unused_target_expr = false;

        let block_id = self.lower_body(|this| {
            let &DelegationResolution { param_info, span, is_method, .. } = res;
            let ParamInfo { param_count, .. } = param_info;
            let arguments_to_map = &res.sig_mapping.arguments_to_map;

            let mut parameters: Vec<hir::Param<'_>> = Vec::with_capacity(param_count);
            let mut args: Vec<hir::Expr<'_>> = Vec::with_capacity(param_count);
            let mut stmts = vec![];

            // Consider non-specified target expression as generated,
            // as we do not want to emit error when target expression is
            // not specified.
            unused_target_expr =
                block.is_some() && (param_count == 0 || arguments_to_map.is_empty());

            for idx in 0..param_count {
                let (param, pat_node_id) = this.generate_param(is_method, idx, span);
                parameters.push(param);

                let generate_arg =
                    |this: &mut Self| this.generate_arg(is_method, idx, param.pat.hir_id, span);

                let arg = block
                    .filter(|_| arguments_to_map.contains(&idx))
                    .and_then(|block| {
                        let block = this.lower_block_maybe_more_than_once(
                            block,
                            pat_node_id,
                            param.pat.hir_id.local_id,
                            delegation.id,
                        );

                        stmts.push(block.stmts);

                        // The behavior of the delegation's target expression differs from the
                        // behavior of the usual block, where if there is no final expression
                        // the `()` is returned. In case of the similar situation in delegation
                        // (no final expression) we propagate first argument instead of replacing
                        // it with `()`.
                        block.expr.copied()
                    })
                    .unwrap_or_else(|| generate_arg(this));

                args.push(arg);
            }

            let (final_expr, hir_id) = this.finalize_body_lowering(
                delegation,
                this.arena.alloc_from_iter(stmts.into_iter().flatten().copied()),
                args,
                res,
                generics,
                span,
            );

            call_expr_id = hir_id;

            (this.arena.alloc_from_iter(parameters), final_expr)
        });

        debug_assert_ne!(call_expr_id, HirId::INVALID);

        (block_id, call_expr_id, unused_target_expr)
    }

    fn lower_block_maybe_more_than_once(
        &mut self,
        block: &Block,
        pat_node_id: NodeId,
        param_local_id: hir::ItemLocalId,
        delegation_id: NodeId,
    ) -> hir::Block<'hir> {
        let mut self_resolver = SelfResolver {
            ctxt: self,
            path_id: delegation_id,
            self_param_id: pat_node_id,
            overwrites: vec![],
        };

        self_resolver.visit_block(block);

        let overwrites = self_resolver.overwrites;

        // Target expr needs to lower `self` path.
        self.ident_and_label_to_local_id.insert(pat_node_id, param_local_id);

        let block = cfg_select! {
            debug_assertions => {
                crate::re_lowering::ReloweringChecker::allow_relowering(self, |this| {
                    this.lower_block_noalloc(HirId::INVALID, block, false)
                })
            },
            _ => self.lower_block_noalloc(HirId::INVALID, block, false)
        };

        // Remove node ids for which we overwrote resolution to generated param
        // before block lowering as block can be relowered. We need to do it because
        // check in `SelfResolver` uses `get_partial_res` to decide whether to overwrite
        // resolution, and if it is already overwritten from previous block lowering this
        // check will not pass.
        for id in overwrites {
            self.partial_res_overrides.remove(&id);
        }

        block
    }

    fn finalize_body_lowering(
        &mut self,
        delegation: &Delegation,
        stmts: &'hir [hir::Stmt<'hir>],
        args: Vec<hir::Expr<'hir>>,
        res: &DelegationResolution,
        generics: &mut GenericsGenerationResults<'hir>,
        span: Span,
    ) -> (hir::Expr<'hir>, HirId) {
        let path = self.lower_qpath(
            delegation.id,
            &delegation.qself,
            &delegation.path,
            ParamMode::Optional,
            AllowReturnTypeNotation::No,
            ImplTraitContext::Disallowed(ImplTraitPosition::Path),
            None,
        );

        let new_path = match path {
            hir::QPath::Resolved(ty, path) => {
                let mut new_path = path.clone();
                let len = new_path.segments.len();

                new_path.segments = self.arena.alloc_from_iter(
                    new_path.segments.iter().enumerate().map(|(idx, segment)| {
                        if idx + 2 == len {
                            self.process_segment(span, segment, &mut generics.parent)
                        } else if idx + 1 == len {
                            self.process_segment(span, segment, &mut generics.child)
                        } else {
                            segment.clone()
                        }
                    }),
                );

                // Explicitly create `Self` self-type in case of infers or static
                // free-to-trait reuses.
                let ty = match generics.self_ty_propagation_kind {
                    Some(hir::DelegationSelfTyPropagationKind::SelfParam) => {
                        let self_param = generics.parent.generics.find_self_param();
                        let path = self.create_generic_arg_path(self_param);
                        let kind = hir::TyKind::Path(path);

                        let ty = match ty {
                            Some(ty) => hir::Ty { kind, ..ty.clone() },
                            None => hir::Ty { kind, hir_id: self.next_id(), span },
                        };

                        Some(&*self.arena.alloc(ty))
                    }
                    _ => ty,
                };

                hir::QPath::Resolved(ty, self.arena.alloc(new_path))
            }
            hir::QPath::TypeRelative(..) => unreachable!("until inherent methods are supported"),
        };

        if let Some(hir::DelegationSelfTyPropagationKind::SelfTy(id)) =
            generics.self_ty_propagation_kind.as_mut()
        {
            *id = match new_path {
                hir::QPath::Resolved(ty, _) => {
                    ty.expect("must contain self type as `SelfTy` propagation kind is specified")
                }
                hir::QPath::TypeRelative(ty, _) => ty,
            }
            .hir_id;
        }

        let callee_path = self.arena.alloc(self.mk_expr(hir::ExprKind::Path(new_path), span));
        let args = self.arena.alloc_from_iter(args);
        let call = self.mk_expr(hir::ExprKind::Call(callee_path, args), span);

        let expr = if res.sig_mapping.map_return {
            let res = Res::SelfTyAlias {
                alias_to: res.parent.to_def_id(),
                is_trait_impl: self.tcx.def_kind(res.parent) == DefKind::Impl { of_trait: true },
            };

            let ident = Ident::new(kw::SelfUpper, span);
            let path = self.create_resolved_path(res, ident, span);

            // FIXME(fn_delegation): add default `..` for all other fields.
            let initializer = hir::ExprKind::Struct(
                self.arena.alloc(path),
                self.arena.alloc_slice(&[hir::ExprField {
                    hir_id: self.next_id(),
                    is_shorthand: false,
                    ident: Ident::new(sym::integer(0), span),
                    expr: self.arena.alloc(call),
                    span,
                }]),
                hir::StructTailExpr::None,
            );

            self.arena.alloc(self.mk_expr(initializer, span))
        } else {
            self.arena.alloc(call)
        };

        let block = self.arena.alloc(hir::Block {
            stmts,
            expr: Some(expr),
            hir_id: self.next_id(),
            rules: hir::BlockCheckMode::DefaultBlock,
            span,
            targeted_by_break: false,
        });

        (self.mk_expr(hir::ExprKind::Block(block, None), span), call.hir_id)
    }

    fn process_segment(
        &mut self,
        span: Span,
        segment: &hir::PathSegment<'hir>,
        result: &mut GenericsGenerationResult<'hir>,
    ) -> hir::PathSegment<'hir> {
        let infer_indices = result.generics.infer_indices();
        result.generics.into_hir_generics(self, span);

        let mut segment = segment.clone();
        let mut args_iter = result.generics.create_args_iterator();

        let new_args = segment
            .args
            .filter(|args| !args.is_empty())
            .map(|args| {
                self.arena.alloc_from_iter(args.args.iter().enumerate().map(|(idx, arg)| {
                    if infer_indices.contains(&idx) {
                        args_iter.next(self, |_| arg.hir_id()).expect("arg must exist for infer")
                    } else {
                        *arg
                    }
                }))
            })
            .unwrap_or_else(|| self.arena.alloc_from_iter(args_iter.consume_all(self)));

        // Do not omit constraints as there might be some and they must be present in HIR (#158812).
        let has_constraints = segment.args.is_some_and(|a| !a.constraints.is_empty());

        // Needed for better error messages (`trait-impl-wrong-args-count.rs` test).
        segment.args = (has_constraints || !new_args.is_empty()).then(|| {
            &*self.arena.alloc(hir::GenericArgs {
                args: new_args,
                constraints: segment.args.map(|a| a.constraints).unwrap_or(&[]),
                parenthesized: hir::GenericArgsParentheses::No,
                span_ext: segment.args.map_or(span, |args| args.span_ext),
            })
        });

        result.args_segment_id = segment.hir_id;
        result.use_for_sig_inheritance = !result.generics.is_trait_impl();

        segment.delegation_child_segment = result.generics.pos() == GenericsPosition::Child;

        segment
    }

    fn generate_delegation_error(
        &mut self,
        span: Span,
        delegation: &Delegation,
    ) -> DelegationResults<'hir> {
        let decl = self.arena.alloc(hir::FnDecl::dummy(span));

        let header = self.generate_header_error();
        let sig = hir::FnSig { decl, header, span };

        let ident = self.lower_ident(delegation.ident);

        let body_id = self.lower_body(|this| {
            let path = this.lower_qpath(
                delegation.id,
                &delegation.qself,
                &delegation.path,
                ParamMode::Optional,
                AllowReturnTypeNotation::No,
                ImplTraitContext::Disallowed(ImplTraitPosition::Path),
                None,
            );

            let callee_path = this.arena.alloc(this.mk_expr(hir::ExprKind::Path(path), span));
            let args = if let Some(block) = &delegation.body {
                this.arena.alloc_slice(&[this.lower_block_expr(block)])
            } else {
                &mut []
            };

            let call = this.arena.alloc(this.mk_expr(hir::ExprKind::Call(callee_path, args), span));

            let block = this.arena.alloc(hir::Block {
                stmts: &[],
                expr: Some(call),
                hir_id: this.next_id(),
                rules: hir::BlockCheckMode::DefaultBlock,
                span,
                targeted_by_break: false,
            });

            (&[], this.mk_expr(hir::ExprKind::Block(block, None), span))
        });

        let generics = hir::Generics::empty();
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

struct SelfResolver<'a, 'b, 'hir> {
    ctxt: &'a mut LoweringContext<'b, 'hir>,
    path_id: NodeId,
    self_param_id: NodeId,
    overwrites: Vec<NodeId>,
}

impl SelfResolver<'_, '_, '_> {
    fn try_replace_id(&mut self, id: NodeId) {
        if let Some(res) = self.ctxt.get_partial_res(id)
            && let Some(Res::Local(sig_id)) = res.full_res()
            && sig_id == self.path_id
        {
            self.overwrites.push(id);
            self.ctxt.partial_res_overrides.insert(id, self.self_param_id);
        }
    }
}

impl<'ast> Visitor<'ast> for SelfResolver<'_, '_, '_> {
    fn visit_id(&mut self, id: NodeId) {
        self.try_replace_id(id);
    }
}
