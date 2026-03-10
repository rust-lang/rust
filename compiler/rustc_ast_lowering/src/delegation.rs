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
use rustc_ast as ast;
use rustc_ast::*;
use rustc_data_structures::fx::FxHashSet;
use rustc_errors::ErrorGuaranteed;
use rustc_hir as hir;
use rustc_hir::attrs::{AttributeKind, InlineAttr};
use rustc_hir::def_id::DefId;
use rustc_middle::span_bug;
use rustc_middle::ty::Asyncness;
use rustc_span::symbol::kw;
use rustc_span::{Ident, Span, Symbol};
use smallvec::SmallVec;

use crate::delegation::generics::{GenericsGenerationResult, GenericsGenerationResults};
use crate::errors::{CycleInDelegationSignatureResolution, UnresolvedDelegationCallee};
use crate::{
    AllowReturnTypeNotation, CombinedResolverForLowering, GenericArgsMode, ImplTraitContext,
    ImplTraitPosition, LoweringContext, ParamMode, ResolverAstLoweringExt,
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

impl<'hir> LoweringContext<'_, '_, 'hir> {
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
        let span = self.lower_span(delegation.path.segments.last().unwrap().ident.span);

        // Delegation can be unresolved in illegal places such as function bodies in extern blocks (see #151356)
        let delegee_id = if let Some(delegation_info) =
            self.resolver.delegation_info(self.local_def_id(item_id))
        {
            self.get_delegee_id(delegation_info.resolution_node, span)
        } else {
            return self.generate_delegation_error(
                self.dcx().span_delayed_bug(
                    span,
                    format!("LoweringContext: the delegation {:?} is unresolved", item_id),
                ),
                span,
                delegation,
            );
        };

        match delegee_id {
            Ok(delegee_id) => {
                self.add_attrs_if_needed(span, delegee_id);

                let is_method = self.is_method(delegee_id, span);

                let (param_count, c_variadic) = self.param_count(delegee_id);

                let mut generics =
                    self.lower_delegation_generics(delegation, delegee_id, item_id, span);

                let body_id = self.lower_delegation_body(
                    delegation,
                    item_id,
                    is_method,
                    param_count,
                    &mut generics,
                    span,
                );

                let decl = self.lower_delegation_decl(
                    delegee_id,
                    param_count,
                    c_variadic,
                    span,
                    &generics,
                );

                let sig = self.lower_delegation_sig(delegee_id, decl, span);
                let ident = self.lower_ident(delegation.ident);

                let generics = self.arena.alloc(hir::Generics {
                    has_where_clause_predicates: false,
                    params: self.arena.alloc_from_iter(generics.all_params(item_id, span, self)),
                    predicates: self
                        .arena
                        .alloc_from_iter(generics.all_predicates(item_id, span, self)),
                    span,
                    where_clause_span: span,
                });

                DelegationResults { body_id, sig, ident, generics }
            }
            Err(err) => self.generate_delegation_error(err, span, delegation),
        }
    }

    fn add_attrs_if_needed(&mut self, span: Span, delegee_id: DefId) {
        let new_attrs =
            self.create_new_attrs(ATTRS_ADDITIONS, span, delegee_id, self.attrs.get(&PARENT_ID));

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
        delegee_id: DefId,
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
                    AttrAdditionKind::Inherit { factory, .. } => {
                        #[allow(deprecated)]
                        let original_attr = self
                            .tcx
                            .get_all_attrs(delegee_id)
                            .iter()
                            .find(|base_attr| (addition_info.equals)(base_attr));

                        if let Some(original_attr) = original_attr {
                            return Some(factory(span, original_attr));
                        }

                        None
                    }
                }
            })
            .collect::<Vec<_>>()
    }

    fn get_delegee_id(&self, mut node_id: NodeId, span: Span) -> Result<DefId, ErrorGuaranteed> {
        let mut visited: FxHashSet<NodeId> = Default::default();
        let mut path: SmallVec<[DefId; 1]> = Default::default();

        loop {
            visited.insert(node_id);

            let Some(def_id) = self.get_resolution_id(node_id) else {
                return Err(self.tcx.dcx().span_delayed_bug(
                    span,
                    format!(
                        "LoweringContext: couldn't resolve node {:?} in delegation item",
                        node_id
                    ),
                ));
            };

            path.push(def_id);

            // If def_id is in local crate and it corresponds to another delegation
            // it means that we refer to another delegation as a callee, so in order to obtain
            // a signature DefId we obtain NodeId of the callee delegation and try to get signature from it.
            if let Some(local_id) = def_id.as_local()
                && let Some(delegation_info) = self.resolver.delegation_info(local_id)
            {
                node_id = delegation_info.resolution_node;
                if visited.contains(&node_id) {
                    // We encountered a cycle in the resolution, or delegation callee refers to non-existent
                    // entity, in this case emit an error.
                    return Err(match visited.len() {
                        1 => self.dcx().emit_err(UnresolvedDelegationCallee { span }),
                        _ => self.dcx().emit_err(CycleInDelegationSignatureResolution { span }),
                    });
                }
            } else {
                return Ok(path[0]);
            }
        }
    }

    fn get_resolution_id(&self, node_id: NodeId) -> Option<DefId> {
        self.resolver.get_partial_res(node_id).and_then(|r| r.expect_full_res().opt_def_id())
    }

    // Function parameter count, including C variadic `...` if present.
    fn param_count(&self, def_id: DefId) -> (usize, bool /*c_variadic*/) {
        let sig = self.tcx.fn_sig(def_id).skip_binder().skip_binder();
        (sig.inputs().len() + usize::from(sig.c_variadic), sig.c_variadic)
    }

    fn lower_delegation_decl(
        &mut self,
        sig_id: DefId,
        param_count: usize,
        c_variadic: bool,
        span: Span,
        generics: &GenericsGenerationResults<'hir>,
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
            kind: hir::TyKind::InferDelegation(
                sig_id,
                hir::InferDelegationKind::Output(self.arena.alloc(hir::DelegationGenerics {
                    child_args_segment_id: generics.child.args_segment_id,
                    parent_args_segment_id: generics.parent.args_segment_id,
                })),
            ),
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
        let sig = self.tcx.fn_sig(sig_id).skip_binder().skip_binder();
        let asyncness = match self.tcx.asyncness(sig_id) {
            Asyncness::Yes => hir::IsAsync::Async(span),
            Asyncness::No => hir::IsAsync::NotAsync,
        };

        let header = hir::FnHeader {
            safety: if self.tcx.codegen_fn_attrs(sig_id).safe_target_features {
                hir::HeaderSafety::SafeTargetFeatures
            } else {
                hir::HeaderSafety::Normal(sig.safety)
            },
            constness: self.tcx.constness(sig_id),
            asyncness,
            abi: sig.abi,
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
        item_id: NodeId,
        is_method: bool,
        param_count: usize,
        generics: &mut GenericsGenerationResults<'hir>,
        span: Span,
    ) -> BodyId {
        let block = delegation.body.as_deref();

        self.lower_body(|this| {
            let mut parameters: Vec<hir::Param<'_>> = Vec::with_capacity(param_count);
            let mut args: Vec<hir::Expr<'_>> = Vec::with_capacity(param_count);

            for idx in 0..param_count {
                let (param, pat_node_id) = this.generate_param(is_method, idx, span);
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
                    this.generate_arg(is_method, idx, param.pat.hir_id, span)
                };
                args.push(arg);
            }

            let final_expr = this.finalize_body_lowering(delegation, item_id, args, generics, span);

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
        item_id: NodeId,
        args: Vec<hir::Expr<'hir>>,
        generics: &mut GenericsGenerationResults<'hir>,
        span: Span,
    ) -> hir::Expr<'hir> {
        let args = self.arena.alloc_from_iter(args);

        let has_generic_args =
            delegation.path.segments.iter().rev().skip(1).any(|segment| segment.args.is_some());

        let call = if self
            .get_resolution_id(delegation.id)
            .map(|def_id| self.is_method(def_id, span))
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

            // FIXME(fn_delegation): proper support for parent generics propagation
            // in method call scenario.
            let segment = self.process_segment(item_id, span, &segment, &mut generics.child, false);
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

            let new_path = match path {
                hir::QPath::Resolved(ty, path) => {
                    let mut new_path = path.clone();
                    let len = new_path.segments.len();

                    new_path.segments = self.arena.alloc_from_iter(
                        new_path.segments.iter().enumerate().map(|(idx, segment)| {
                            let mut process_segment = |result, add_lifetimes| {
                                self.process_segment(item_id, span, segment, result, add_lifetimes)
                            };

                            if idx + 2 == len {
                                process_segment(&mut generics.parent, true)
                            } else if idx + 1 == len {
                                process_segment(&mut generics.child, false)
                            } else {
                                segment.clone()
                            }
                        }),
                    );

                    hir::QPath::Resolved(ty, self.arena.alloc(new_path))
                }
                hir::QPath::TypeRelative(ty, segment) => {
                    let segment =
                        self.process_segment(item_id, span, segment, &mut generics.child, false);

                    hir::QPath::TypeRelative(ty, self.arena.alloc(segment))
                }
            };

            let callee_path = self.arena.alloc(self.mk_expr(hir::ExprKind::Path(new_path), span));
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

    fn process_segment(
        &mut self,
        item_id: NodeId,
        span: Span,
        segment: &hir::PathSegment<'hir>,
        result: &mut GenericsGenerationResult<'hir>,
        add_lifetimes: bool,
    ) -> hir::PathSegment<'hir> {
        let details = result.generics.args_propagation_details();

        // The first condition is needed when there is SelfAndUserSpecified case,
        // we don't want to propagate generics params in this situation.
        let segment = if details.should_propagate
            && let Some(args) = result
                .generics
                .into_hir_generics(self, item_id, span)
                .into_generic_args(self, add_lifetimes, span)
        {
            hir::PathSegment { args: Some(args), ..segment.clone() }
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
        err: ErrorGuaranteed,
        span: Span,
        delegation: &Delegation,
    ) -> DelegationResults<'hir> {
        let decl = self.arena.alloc(hir::FnDecl {
            inputs: &[],
            output: hir::FnRetTy::DefaultReturn(span),
            c_variadic: false,
            lifetime_elision_allowed: true,
            implicit_self: hir::ImplicitSelfKind::None,
        });

        let header = self.generate_header_error();
        let sig = hir::FnSig { decl, header, span };

        let ident = self.lower_ident(delegation.ident);

        let body_id = self.lower_body(|this| {
            let body_expr = match delegation.body.as_ref() {
                Some(box block) => {
                    // Generates a block when we failed to resolve delegation, where a target expression is its only statement,
                    // thus there will be no ICEs on further stages of analysis (see #144594)

                    // As we generate a void function we want to convert target expression to statement to avoid additional
                    // errors, such as mismatched return type
                    let stmts = this.arena.alloc_from_iter([hir::Stmt {
                        hir_id: this.next_id(),
                        kind: rustc_hir::StmtKind::Semi(
                            this.arena.alloc(this.lower_target_expr(block)),
                        ),
                        span,
                    }]);

                    let block = this.arena.alloc(hir::Block {
                        stmts,
                        expr: None,
                        hir_id: this.next_id(),
                        rules: hir::BlockCheckMode::DefaultBlock,
                        span,
                        targeted_by_break: false,
                    });

                    hir::ExprKind::Block(block, None)
                }
                None => hir::ExprKind::Err(err),
            };

            (&[], this.mk_expr(body_expr, span))
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
    resolver: &'b mut CombinedResolverForLowering<'a, 'hir>,
    path_id: NodeId,
    self_param_id: NodeId,
}

impl<'a, 'b, 'hir> SelfResolver<'a, 'b, 'hir> {
    fn try_replace_id(&mut self, id: NodeId) {
        if let Some(res) = self.resolver.get_partial_res(id)
            && let Some(Res::Local(sig_id)) = res.full_res()
            && sig_id == self.path_id
        {
            let new_res = PartialRes::new(Res::Local(self.self_param_id));
            self.resolver.mut_part.partial_res_map.insert(id, new_res);
        }
    }
}

impl<'ast, 'a> Visitor<'ast> for SelfResolver<'a, '_, '_> {
    fn visit_id(&mut self, id: NodeId) {
        self.try_replace_id(id);
    }
}
