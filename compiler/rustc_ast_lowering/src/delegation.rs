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
use std::ops::ControlFlow;

use ast::visit::Visitor;
use hir::def::{DefKind, Res};
use hir::{BodyId, HirId};
use rustc_abi::ExternAbi;
use rustc_ast as ast;
use rustc_ast::node_id::NodeMap;
use rustc_ast::*;
use rustc_data_structures::fx::{FxHashSet, FxIndexMap};
use rustc_hir::attrs::{AttributeKind, InlineAttr};
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_hir::{self as hir, FnDeclFlags};
use rustc_middle::span_bug;
use rustc_middle::ty::{Asyncness, PerOwnerResolverData, TyCtxt};
use rustc_span::symbol::kw;
use rustc_span::{ErrorGuaranteed, Ident, Span, Symbol};

use crate::delegation::generics::{GenericsGenerationResult, GenericsGenerationResults};
use crate::diagnostics::{
    CycleInDelegationSignatureResolution, DelegationAttemptedBlockWithDefsDeletion,
    DelegationBlockSpecifiedWhenNoParams, UnresolvedDelegationCallee,
};
use crate::{
    AllowReturnTypeNotation, ImplTraitContext, ImplTraitPosition, LoweringContext, ParamMode,
    index_crate,
};

mod generics;

pub(crate) struct DelegationResults<'hir> {
    pub body_id: hir::BodyId,
    pub sig: hir::FnSig<'hir>,
    pub ident: Ident,
    pub generics: &'hir hir::Generics<'hir>,
}

struct AttrAdditionInfo {
    pub equals: fn(&hir::Attribute) -> bool,
    pub kind: AttrAdditionKind,
}

enum AttrAdditionKind {
    Default { factory: fn(Span) -> hir::Attribute },
    Inherit { factory: fn(Span, &hir::Attribute) -> hir::Attribute },
}

/// Summary info about function parameters.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
struct ParamInfo {
    /// The number of function parameters, including any C variadic `...` parameter.
    pub param_count: usize,

    /// Whether the function arguments end in a C variadic `...` parameter.
    pub c_variadic: bool,

    /// The index of the splatted parameter, if any.
    pub splatted: Option<u16>,
}

const PARENT_ID: hir::ItemLocalId = hir::ItemLocalId::ZERO;

static ATTRS_ADDITIONS: &[AttrAdditionInfo] = &[
    AttrAdditionInfo {
        equals: |a| matches!(a, hir::Attribute::Parsed(AttributeKind::MustUse { .. })),
        kind: AttrAdditionKind::Inherit {
            factory: |span, original_attr| {
                let reason = match original_attr {
                    hir::Attribute::Parsed(AttributeKind::MustUse { reason, .. }) => *reason,
                    _ => None,
                };

                hir::Attribute::Parsed(AttributeKind::MustUse { span, reason })
            },
        },
    },
    AttrAdditionInfo {
        equals: |a| matches!(a, hir::Attribute::Parsed(AttributeKind::Inline(..))),
        kind: AttrAdditionKind::Default {
            factory: |span| hir::Attribute::Parsed(AttributeKind::Inline(InlineAttr::Hint, span)),
        },
    },
];

pub(crate) fn delegations_resolutions(
    tcx: TyCtxt<'_>,
    _: (),
) -> FxIndexMap<LocalDefId, Result<DefId, ErrorGuaranteed>> {
    let krate = tcx.hir_crate(());

    let (resolver, ast_crate) = &*krate.delayed_resolver.borrow();

    // FIXME!!!(fn_delegation): make ast index lifetime same as resolver,
    // as it is too bad to reindex whole crate on each delegation lowering.
    let ast_index = index_crate(resolver, ast_crate);

    let mut result = FxIndexMap::<LocalDefId, Result<DefId, ErrorGuaranteed>>::default();

    for &def_id in &krate.delayed_ids {
        let delegation = ast_index[def_id].delegation().expect("processing delegations");
        let span = delegation.last_segment_span();

        if let Some(info) = tcx.resolutions(()).delegation_infos.get(&def_id) {
            let res = info.resolution_id.map(|id| check_for_cycles(tcx, id, span).map(|_| id));
            result.insert(def_id, res.flatten());
        } else {
            tcx.dcx().span_delayed_bug(
                span,
                format!("delegation resolution record was not found for {def_id:?}"),
            );
        }
    }

    result
}

fn check_for_cycles(tcx: TyCtxt<'_>, mut def_id: DefId, span: Span) -> Result<(), ErrorGuaranteed> {
    let mut visited: FxHashSet<DefId> = Default::default();

    loop {
        visited.insert(def_id);

        // If def_id is in local crate and it corresponds to another delegation
        // it means that we refer to another delegation as a callee, so in order to obtain
        // a signature DefId we obtain NodeId of the callee delegation and try to get signature from it.
        if let Some(local_id) = def_id.as_local()
            && let Some(info) = tcx.resolutions(()).delegation_infos.get(&local_id)
            && let Ok(id) = info.resolution_id
        {
            def_id = id;
            if visited.contains(&def_id) {
                return Err(match visited.len() {
                    1 => tcx.dcx().emit_err(UnresolvedDelegationCallee { span }),
                    _ => tcx.dcx().emit_err(CycleInDelegationSignatureResolution { span }),
                });
            }
        } else {
            return Ok(());
        }
    }
}

impl<'hir> LoweringContext<'_, 'hir> {
    fn is_method(&self, def_id: DefId, span: Span) -> bool {
        match self.tcx.def_kind(def_id) {
            DefKind::Fn => false,
            DefKind::AssocFn => self.tcx.associated_item(def_id).is_method(),
            _ => span_bug!(span, "unexpected DefKind for delegation item"),
        }
    }

    pub(crate) fn lower_delegation(
        &mut self,
        delegation: &Delegation,
        item_id: NodeId,
    ) -> DelegationResults<'hir> {
        let span = self.lower_span(delegation.last_segment_span());

        let sig_id = self.tcx.delegations_resolutions(()).get(&self.owner.def_id).copied();

        // Delegation can be missing from the `delegations_resolutions` table
        // in illegal places such as function bodies in extern blocks (see #151356).
        let Some(Ok(sig_id)) = sig_id else {
            self.dcx().span_delayed_bug(
                span,
                format!("LoweringContext: the delegation {:?} is unresolved", item_id),
            );

            return self.generate_delegation_error(span, delegation);
        };

        self.add_attrs_if_needed(span, sig_id);

        let is_method = self.is_method(sig_id, span);

        let param_info = self.param_info(sig_id);

        if !self.check_block_soundness(delegation, sig_id, is_method, param_info.param_count) {
            return self.generate_delegation_error(span, delegation);
        }

        let mut generics = self.uplift_delegation_generics(delegation, sig_id, is_method);

        let (body_id, call_expr_id, unused_target_expr) = self.lower_delegation_body(
            delegation,
            sig_id,
            param_info.param_count,
            &mut generics,
            span,
        );

        let decl = self.lower_delegation_decl(
            delegation.source,
            sig_id,
            param_info,
            span,
            &generics,
            delegation.id,
            call_expr_id,
            unused_target_expr,
        );

        let sig = self.lower_delegation_sig(sig_id, decl, span);
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

    fn check_block_soundness(
        &self,
        delegation: &Delegation,
        sig_id: DefId,
        is_method: bool,
        param_count: usize,
    ) -> bool {
        let Some(block) = delegation.body.as_ref() else { return true };
        let should_generate_block = self.should_generate_block(delegation, sig_id, is_method);

        // Report an error if user has explicitly specified delegation's target expression
        // in a single delegation when reused function has no params.
        if param_count == 0 && should_generate_block {
            self.dcx().emit_err(DelegationBlockSpecifiedWhenNoParams { span: block.span });
            return false;
        }

        struct DefinitionsFinder<'a> {
            all_owners: &'a NodeMap<PerOwnerResolverData<'a>>,
            // `self.owner.node_id_to_def_id`
            nested_def_ids: &'a NodeMap<LocalDefId>,
        }

        impl<'a> ast::visit::Visitor<'a> for DefinitionsFinder<'a> {
            type Result = ControlFlow<()>;

            fn visit_id(&mut self, id: NodeId) -> Self::Result {
                /*
                    (from `tests\ui\delegation\target-expr-removal-defs-inside.rs`):
                    ```rust
                        reuse impl Trait for S1 {
                            some::path::<{ fn foo() {} }>::xd();
                            fn foo() {}
                            self.0
                        }
                    ```

                    Constant from unresolved path will be in `nested_owners`,
                    `fn foo() {}` will not be in `nested_owners` but will be in `owners`,
                    both have `LocalDefId`, so we check those two maps.
                */
                match self.all_owners.contains_key(&id) || self.nested_def_ids.contains_key(&id) {
                    true => ControlFlow::Break(()),
                    false => ControlFlow::Continue(()),
                }
            }
        }

        let mut collector = DefinitionsFinder {
            all_owners: &self.resolver.owners,
            nested_def_ids: &self.owner.node_id_to_def_id,
        };

        let contains_defs = collector.visit_block(block).is_break();

        // If there are definitions inside and we can't delete target expression, so report an error.
        // FIXME(fn_delegation): support deletion of target expression with defs inside.
        if !should_generate_block && contains_defs {
            self.dcx().emit_err(DelegationAttemptedBlockWithDefsDeletion { span: block.span });
            return false;
        }

        true
    }

    fn should_generate_block(
        &self,
        delegation: &Delegation,
        sig_id: DefId,
        is_method: bool,
    ) -> bool {
        is_method
            || matches!(self.tcx.def_kind(sig_id), DefKind::Fn)
            || matches!(delegation.source, DelegationSource::Single)
    }

    fn add_attrs_if_needed(&mut self, span: Span, sig_id: DefId) {
        let new_attrs =
            self.create_new_attrs(ATTRS_ADDITIONS, span, sig_id, self.attrs.get(&PARENT_ID));

        if new_attrs.is_empty() {
            return;
        }

        let new_arena_allocated_attrs = match self.attrs.get(&PARENT_ID) {
            Some(existing_attrs) => self.arena.alloc_from_iter(
                existing_attrs.iter().map(|a| a.clone()).chain(new_attrs.into_iter()),
            ),
            None => self.arena.alloc_from_iter(new_attrs.into_iter()),
        };

        self.attrs.insert(PARENT_ID, new_arena_allocated_attrs);
    }

    fn create_new_attrs(
        &self,
        candidate_additions: &[AttrAdditionInfo],
        span: Span,
        sig_id: DefId,
        existing_attrs: Option<&&[hir::Attribute]>,
    ) -> Vec<hir::Attribute> {
        candidate_additions
            .iter()
            .filter_map(|addition_info| {
                if let Some(existing_attrs) = existing_attrs
                    && existing_attrs
                        .iter()
                        .any(|existing_attr| (addition_info.equals)(existing_attr))
                {
                    return None;
                }

                match addition_info.kind {
                    AttrAdditionKind::Default { factory } => Some(factory(span)),
                    AttrAdditionKind::Inherit { factory, .. } =>
                    {
                        #[allow(deprecated)]
                        self.tcx
                            .get_all_attrs(sig_id)
                            .iter()
                            .find_map(|a| (addition_info.equals)(a).then(|| factory(span, a)))
                    }
                }
            })
            .collect::<Vec<_>>()
    }

    fn get_resolution_id(&self, node_id: NodeId) -> Option<DefId> {
        self.get_partial_res(node_id).and_then(|r| r.expect_full_res().opt_def_id())
    }

    /// Returns function parameter info, including C variadic `...` and `#[splat]` if present.
    fn param_info(&self, def_id: DefId) -> ParamInfo {
        let sig = self.tcx.fn_sig(def_id).skip_binder().skip_binder();

        // FIXME(splat): use `sig.splatted()` once FnSig has it
        ParamInfo {
            param_count: sig.inputs().len() + usize::from(sig.c_variadic()),
            c_variadic: sig.c_variadic(),
            splatted: None,
        }
    }

    fn lower_delegation_decl(
        &mut self,
        source: DelegationSource,
        sig_id: DefId,
        param_info: ParamInfo,
        span: Span,
        generics: &GenericsGenerationResults<'hir>,
        call_path_node_id: NodeId,
        call_expr_id: HirId,
        unused_target_expr: bool,
    ) -> &'hir hir::FnDecl<'hir> {
        let ParamInfo { param_count, c_variadic, splatted } = param_info;

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
                    call_path_res: self.get_resolution_id(call_path_node_id),
                    child_args_segment_id: generics.child.args_segment_id,
                    parent_args_segment_id: generics.parent.args_segment_id,
                    self_ty_id: generics.self_ty_id,
                    propagate_self_ty: generics.propagate_self_ty,
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
        }));

        let path = self.arena.alloc(hir::Path { span, res: Res::Local(param_id), segments });
        self.mk_expr(hir::ExprKind::Path(hir::QPath::Resolved(None, path)), span)
    }

    fn lower_delegation_body(
        &mut self,
        delegation: &Delegation,
        sig_id: DefId,
        param_count: usize,
        generics: &mut GenericsGenerationResults<'hir>,
        span: Span,
    ) -> (BodyId, HirId, bool) {
        let block = delegation.body.as_deref();
        let mut call_expr_id = HirId::INVALID;
        let mut unused_target_expr = false;

        let block_id = self.lower_body(|this| {
            let mut parameters: Vec<hir::Param<'_>> = Vec::with_capacity(param_count);
            let mut args: Vec<hir::Expr<'_>> = Vec::with_capacity(param_count);
            let mut stmts: &[hir::Stmt<'hir>] = &[];

            let is_method = this.is_method(sig_id, span);
            let should_generate_block = this.should_generate_block(delegation, sig_id, is_method);

            // Consider non-specified target expression as generated,
            // as we do not want to emit error when target expression is
            // not specified.
            unused_target_expr = block.is_some() && (param_count == 0 || !should_generate_block);

            for idx in 0..param_count {
                let (param, pat_node_id) = this.generate_param(is_method, idx, span);
                parameters.push(param);

                let generate_arg =
                    |this: &mut Self| this.generate_arg(is_method, idx, param.pat.hir_id, span);

                let arg = if let Some(block) = block
                    && idx == 0
                    && should_generate_block
                {
                    let mut self_resolver = SelfResolver {
                        ctxt: this,
                        path_id: delegation.id,
                        self_param_id: pat_node_id,
                    };
                    self_resolver.visit_block(block);
                    // Target expr needs to lower `self` path.
                    this.ident_and_label_to_local_id.insert(pat_node_id, param.pat.hir_id.local_id);

                    // Lower with `HirId::INVALID` as we will use only expr and stmts.
                    // FIXME(fn_delegation): Alternatives for target expression lowering:
                    // https://github.com/rust-lang/rfcs/pull/3530#issuecomment-2197170600.
                    let block = this.lower_block_noalloc(HirId::INVALID, block, false);

                    stmts = block.stmts;

                    // The behavior of the delegation's target expression differs from the
                    // behavior of the usual block, where if there is no final expression
                    // the `()` is returned. In case of the similar situation in delegation
                    // (no final expression) we propagate first argument instead of replacing
                    // it with `()`.
                    if let Some(&expr) = block.expr { expr } else { generate_arg(this) }
                } else {
                    generate_arg(this)
                };

                args.push(arg);
            }

            let (final_expr, hir_id) =
                this.finalize_body_lowering(delegation, stmts, args, generics, span);

            call_expr_id = hir_id;

            (this.arena.alloc_from_iter(parameters), final_expr)
        });

        debug_assert_ne!(call_expr_id, HirId::INVALID);

        (block_id, call_expr_id, unused_target_expr)
    }

    fn finalize_body_lowering(
        &mut self,
        delegation: &Delegation,
        stmts: &'hir [hir::Stmt<'hir>],
        args: Vec<hir::Expr<'hir>>,
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

                hir::QPath::Resolved(ty, self.arena.alloc(new_path))
            }
            hir::QPath::TypeRelative(ty, segment) => {
                let segment = self.process_segment(span, segment, &mut generics.child);

                hir::QPath::TypeRelative(ty, self.arena.alloc(segment))
            }
        };

        generics.self_ty_id = match new_path {
            hir::QPath::Resolved(ty, _) => ty,
            hir::QPath::TypeRelative(ty, _) => Some(ty),
        }
        .map(|ty| ty.hir_id);

        let callee_path = self.arena.alloc(self.mk_expr(hir::ExprKind::Path(new_path), span));
        let args = self.arena.alloc_from_iter(args);
        let call = self.arena.alloc(self.mk_expr(hir::ExprKind::Call(callee_path, args), span));

        let block = self.arena.alloc(hir::Block {
            stmts,
            expr: Some(call),
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
        let details = result.generics.args_propagation_details();

        // Always uplift generic params, because if they are not empty then they
        // should be generated in delegation.
        let generics = result.generics.into_hir_generics(self, span);
        let segment = if details.should_propagate {
            let args = generics.into_generic_args(self, span);

            // Needed for better error messages (`trait-impl-wrong-args-count.rs` test).
            let args = if args.is_empty() { None } else { Some(args) };

            hir::PathSegment { args, ..segment.clone() }
        } else {
            segment.clone()
        };

        if details.use_args_in_sig_inheritance {
            result.args_segment_id = Some(segment.hir_id);
        }

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
            let args = if let Some(block) = delegation.body.as_ref() {
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
}

impl SelfResolver<'_, '_, '_> {
    fn try_replace_id(&mut self, id: NodeId) {
        if let Some(res) = self.ctxt.get_partial_res(id)
            && let Some(Res::Local(sig_id)) = res.full_res()
            && sig_id == self.path_id
        {
            self.ctxt.partial_res_overrides.insert(id, self.self_param_id);
        }
    }
}

impl<'ast> Visitor<'ast> for SelfResolver<'_, '_, '_> {
    fn visit_id(&mut self, id: NodeId) {
        self.try_replace_id(id);
    }
}
