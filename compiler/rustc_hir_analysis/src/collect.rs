//! "Collection" is the process of determining the type and other external
//! details of each item in Rust. Collection is specifically concerned
//! with *inter-procedural* things -- for example, for a function
//! definition, collection will figure out the type and signature of the
//! function, but it will not visit the *body* of the function in any way,
//! nor examine type annotations on local variables (that's the job of
//! type *checking*).
//!
//! Collecting is ultimately defined by a bundle of queries that
//! inquire after various facts about the items in the crate (e.g.,
//! `type_of`, `generics_of`, `predicates_of`, etc). See the `provide` function
//! for the full set.
//!
//! At present, however, we do run collection across all items in the
//! crate as a kind of pass. This should eventually be factored away.

use std::cell::Cell;
use std::iter;
use std::ops::Bound;

use rustc_ast::Recovered;
use rustc_data_structures::captures::Captures;
use rustc_data_structures::fx::{FxHashSet, FxIndexMap};
use rustc_data_structures::unord::UnordMap;
use rustc_errors::{
    Applicability, Diag, DiagCtxtHandle, E0228, ErrorGuaranteed, StashKey, struct_span_code_err,
};
use rustc_hir::def::DefKind;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_hir::intravisit::{self, Visitor, walk_generics};
use rustc_hir::{self as hir, GenericParamKind, Node};
use rustc_infer::infer::{InferCtxt, TyCtxtInferExt};
use rustc_infer::traits::ObligationCause;
use rustc_middle::hir::nested_filter;
use rustc_middle::query::Providers;
use rustc_middle::ty::util::{Discr, IntTypeExt};
use rustc_middle::ty::{self, AdtKind, Const, IsSuggestable, Ty, TyCtxt};
use rustc_middle::{bug, span_bug};
use rustc_span::symbol::{Ident, Symbol, kw, sym};
use rustc_span::{DUMMY_SP, Span};
use rustc_target::spec::abi;
use rustc_trait_selection::error_reporting::traits::suggestions::NextTypeParamName;
use rustc_trait_selection::infer::InferCtxtExt;
use rustc_trait_selection::traits::ObligationCtxt;
use tracing::{debug, instrument};

use crate::check::intrinsic::intrinsic_operation_unsafety;
use crate::errors;
use crate::hir_ty_lowering::{HirTyLowerer, RegionInferReason};

pub(crate) mod dump;
mod generics_of;
mod item_bounds;
mod predicates_of;
mod resolve_bound_vars;
mod type_of;

///////////////////////////////////////////////////////////////////////////

pub fn provide(providers: &mut Providers) {
    resolve_bound_vars::provide(providers);
    *providers = Providers {
        type_of: type_of::type_of,
        type_of_opaque: type_of::type_of_opaque,
        type_alias_is_lazy: type_of::type_alias_is_lazy,
        item_bounds: item_bounds::item_bounds,
        explicit_item_bounds: item_bounds::explicit_item_bounds,
        item_super_predicates: item_bounds::item_super_predicates,
        explicit_item_super_predicates: item_bounds::explicit_item_super_predicates,
        item_non_self_assumptions: item_bounds::item_non_self_assumptions,
        impl_super_outlives: item_bounds::impl_super_outlives,
        generics_of: generics_of::generics_of,
        predicates_of: predicates_of::predicates_of,
        explicit_predicates_of: predicates_of::explicit_predicates_of,
        explicit_super_predicates_of: predicates_of::explicit_super_predicates_of,
        explicit_implied_predicates_of: predicates_of::explicit_implied_predicates_of,
        explicit_supertraits_containing_assoc_item:
            predicates_of::explicit_supertraits_containing_assoc_item,
        trait_explicit_predicates_and_bounds: predicates_of::trait_explicit_predicates_and_bounds,
        type_param_predicates: predicates_of::type_param_predicates,
        trait_def,
        adt_def,
        fn_sig,
        impl_trait_header,
        coroutine_kind,
        coroutine_for_closure,
        is_type_alias_impl_trait,
        rendered_precise_capturing_args,
        ..*providers
    };
}

///////////////////////////////////////////////////////////////////////////

/// Context specific to some particular item. This is what implements [`HirTyLowerer`].
///
/// # `ItemCtxt` vs `FnCtxt`
///
/// `ItemCtxt` is primarily used to type-check item signatures and lower them
/// from HIR to their [`ty::Ty`] representation, which is exposed using [`HirTyLowerer`].
/// It's also used for the bodies of items like structs where the body (the fields)
/// are just signatures.
///
/// This is in contrast to `FnCtxt`, which is used to type-check bodies of
/// functions, closures, and `const`s -- anywhere that expressions and statements show up.
///
/// An important thing to note is that `ItemCtxt` does no inference -- it has no [`InferCtxt`] --
/// while `FnCtxt` does do inference.
///
/// [`InferCtxt`]: rustc_infer::infer::InferCtxt
///
/// # Trait predicates
///
/// `ItemCtxt` has information about the predicates that are defined
/// on the trait. Unfortunately, this predicate information is
/// available in various different forms at various points in the
/// process. So we can't just store a pointer to e.g., the HIR or the
/// parsed ty form, we have to be more flexible. To this end, the
/// `ItemCtxt` is parameterized by a `DefId` that it uses to satisfy
/// `probe_ty_param_bounds` requests, drawing the information from
/// the HIR (`hir::Generics`), recursively.
pub struct ItemCtxt<'tcx> {
    tcx: TyCtxt<'tcx>,
    item_def_id: LocalDefId,
    tainted_by_errors: Cell<Option<ErrorGuaranteed>>,
}

///////////////////////////////////////////////////////////////////////////

#[derive(Default)]
pub(crate) struct HirPlaceholderCollector(pub(crate) Vec<Span>);

impl<'v> Visitor<'v> for HirPlaceholderCollector {
    fn visit_ty(&mut self, t: &'v hir::Ty<'v>) {
        if let hir::TyKind::Infer = t.kind {
            self.0.push(t.span);
        }
        intravisit::walk_ty(self, t)
    }
    fn visit_generic_arg(&mut self, generic_arg: &'v hir::GenericArg<'v>) {
        match generic_arg {
            hir::GenericArg::Infer(inf) => {
                self.0.push(inf.span);
                intravisit::walk_inf(self, inf);
            }
            hir::GenericArg::Type(t) => self.visit_ty(t),
            _ => {}
        }
    }
    fn visit_array_length(&mut self, length: &'v hir::ArrayLen<'v>) {
        if let hir::ArrayLen::Infer(inf) = length {
            self.0.push(inf.span);
        }
        intravisit::walk_array_len(self, length)
    }
}

pub struct CollectItemTypesVisitor<'tcx> {
    pub tcx: TyCtxt<'tcx>,
}

/// If there are any placeholder types (`_`), emit an error explaining that this is not allowed
/// and suggest adding type parameters in the appropriate place, taking into consideration any and
/// all already existing generic type parameters to avoid suggesting a name that is already in use.
pub(crate) fn placeholder_type_error<'tcx>(
    cx: &dyn HirTyLowerer<'tcx>,
    generics: Option<&hir::Generics<'_>>,
    placeholder_types: Vec<Span>,
    suggest: bool,
    hir_ty: Option<&hir::Ty<'_>>,
    kind: &'static str,
) {
    if placeholder_types.is_empty() {
        return;
    }

    placeholder_type_error_diag(cx, generics, placeholder_types, vec![], suggest, hir_ty, kind)
        .emit();
}

pub(crate) fn placeholder_type_error_diag<'cx, 'tcx>(
    cx: &'cx dyn HirTyLowerer<'tcx>,
    generics: Option<&hir::Generics<'_>>,
    placeholder_types: Vec<Span>,
    additional_spans: Vec<Span>,
    suggest: bool,
    hir_ty: Option<&hir::Ty<'_>>,
    kind: &'static str,
) -> Diag<'cx> {
    if placeholder_types.is_empty() {
        return bad_placeholder(cx, additional_spans, kind);
    }

    let params = generics.map(|g| g.params).unwrap_or_default();
    let type_name = params.next_type_param_name(None);
    let mut sugg: Vec<_> =
        placeholder_types.iter().map(|sp| (*sp, (*type_name).to_string())).collect();

    if let Some(generics) = generics {
        if let Some(arg) = params.iter().find(|arg| {
            matches!(arg.name, hir::ParamName::Plain(Ident { name: kw::Underscore, .. }))
        }) {
            // Account for `_` already present in cases like `struct S<_>(_);` and suggest
            // `struct S<T>(T);` instead of `struct S<_, T>(T);`.
            sugg.push((arg.span, (*type_name).to_string()));
        } else if let Some(span) = generics.span_for_param_suggestion() {
            // Account for bounds, we want `fn foo<T: E, K>(_: K)` not `fn foo<T, K: E>(_: K)`.
            sugg.push((span, format!(", {type_name}")));
        } else {
            sugg.push((generics.span, format!("<{type_name}>")));
        }
    }

    let mut err =
        bad_placeholder(cx, placeholder_types.into_iter().chain(additional_spans).collect(), kind);

    // Suggest, but only if it is not a function in const or static
    if suggest {
        let mut is_fn = false;
        let mut is_const_or_static = false;

        if let Some(hir_ty) = hir_ty
            && let hir::TyKind::BareFn(_) = hir_ty.kind
        {
            is_fn = true;

            // Check if parent is const or static
            is_const_or_static = matches!(
                cx.tcx().parent_hir_node(hir_ty.hir_id),
                Node::Item(&hir::Item {
                    kind: hir::ItemKind::Const(..) | hir::ItemKind::Static(..),
                    ..
                }) | Node::TraitItem(&hir::TraitItem { kind: hir::TraitItemKind::Const(..), .. })
                    | Node::ImplItem(&hir::ImplItem { kind: hir::ImplItemKind::Const(..), .. })
            );
        }

        // if function is wrapped around a const or static,
        // then don't show the suggestion
        if !(is_fn && is_const_or_static) {
            err.multipart_suggestion(
                "use type parameters instead",
                sugg,
                Applicability::HasPlaceholders,
            );
        }
    }

    err
}

fn reject_placeholder_type_signatures_in_item<'tcx>(
    tcx: TyCtxt<'tcx>,
    item: &'tcx hir::Item<'tcx>,
) {
    let (generics, suggest) = match &item.kind {
        hir::ItemKind::Union(_, generics)
        | hir::ItemKind::Enum(_, generics)
        | hir::ItemKind::TraitAlias(generics, _)
        | hir::ItemKind::Trait(_, _, generics, ..)
        | hir::ItemKind::Impl(hir::Impl { generics, .. })
        | hir::ItemKind::Struct(_, generics) => (generics, true),
        hir::ItemKind::TyAlias(_, generics) => (generics, false),
        // `static`, `fn` and `const` are handled elsewhere to suggest appropriate type.
        _ => return,
    };

    let mut visitor = HirPlaceholderCollector::default();
    visitor.visit_item(item);

    let icx = ItemCtxt::new(tcx, item.owner_id.def_id);

    placeholder_type_error(
        icx.lowerer(),
        Some(generics),
        visitor.0,
        suggest,
        None,
        item.kind.descr(),
    );
}

impl<'tcx> Visitor<'tcx> for CollectItemTypesVisitor<'tcx> {
    type NestedFilter = nested_filter::OnlyBodies;

    fn nested_visit_map(&mut self) -> Self::Map {
        self.tcx.hir()
    }

    fn visit_item(&mut self, item: &'tcx hir::Item<'tcx>) {
        lower_item(self.tcx, item.item_id());
        reject_placeholder_type_signatures_in_item(self.tcx, item);
        intravisit::walk_item(self, item);
    }

    fn visit_generics(&mut self, generics: &'tcx hir::Generics<'tcx>) {
        for param in generics.params {
            match param.kind {
                hir::GenericParamKind::Lifetime { .. } => {}
                hir::GenericParamKind::Type { default: Some(_), .. } => {
                    self.tcx.ensure().type_of(param.def_id);
                }
                hir::GenericParamKind::Type { .. } => {}
                hir::GenericParamKind::Const { default, .. } => {
                    self.tcx.ensure().type_of(param.def_id);
                    if let Some(default) = default {
                        // need to store default and type of default
                        if let hir::ConstArgKind::Anon(ac) = default.kind {
                            self.tcx.ensure().type_of(ac.def_id);
                        }
                        self.tcx.ensure().const_param_default(param.def_id);
                    }
                }
            }
        }
        intravisit::walk_generics(self, generics);
    }

    fn visit_expr(&mut self, expr: &'tcx hir::Expr<'tcx>) {
        if let hir::ExprKind::Closure(closure) = expr.kind {
            self.tcx.ensure().generics_of(closure.def_id);
            self.tcx.ensure().codegen_fn_attrs(closure.def_id);
            // We do not call `type_of` for closures here as that
            // depends on typecheck and would therefore hide
            // any further errors in case one typeck fails.
        }
        intravisit::walk_expr(self, expr);
    }

    /// Don't call `type_of` on opaque types, since that depends on type checking function bodies.
    /// `check_item_type` ensures that it's called instead.
    fn visit_opaque_ty(&mut self, opaque: &'tcx hir::OpaqueTy<'tcx>) {
        let def_id = opaque.def_id;
        self.tcx.ensure().generics_of(def_id);
        self.tcx.ensure().predicates_of(def_id);
        self.tcx.ensure().explicit_item_bounds(def_id);
        self.tcx.ensure().explicit_item_super_predicates(def_id);
        self.tcx.ensure().item_bounds(def_id);
        self.tcx.ensure().item_super_predicates(def_id);
        intravisit::walk_opaque_ty(self, opaque);
    }

    fn visit_trait_item(&mut self, trait_item: &'tcx hir::TraitItem<'tcx>) {
        lower_trait_item(self.tcx, trait_item.trait_item_id());
        intravisit::walk_trait_item(self, trait_item);
    }

    fn visit_impl_item(&mut self, impl_item: &'tcx hir::ImplItem<'tcx>) {
        lower_impl_item(self.tcx, impl_item.impl_item_id());
        intravisit::walk_impl_item(self, impl_item);
    }
}

///////////////////////////////////////////////////////////////////////////
// Utility types and common code for the above passes.

fn bad_placeholder<'cx, 'tcx>(
    cx: &'cx dyn HirTyLowerer<'tcx>,
    mut spans: Vec<Span>,
    kind: &'static str,
) -> Diag<'cx> {
    let kind = if kind.ends_with('s') { format!("{kind}es") } else { format!("{kind}s") };

    spans.sort();
    cx.dcx().create_err(errors::PlaceholderNotAllowedItemSignatures { spans, kind })
}

impl<'tcx> ItemCtxt<'tcx> {
    pub fn new(tcx: TyCtxt<'tcx>, item_def_id: LocalDefId) -> ItemCtxt<'tcx> {
        ItemCtxt { tcx, item_def_id, tainted_by_errors: Cell::new(None) }
    }

    pub fn lower_ty(&self, hir_ty: &hir::Ty<'tcx>) -> Ty<'tcx> {
        self.lowerer().lower_ty(hir_ty)
    }

    pub fn hir_id(&self) -> hir::HirId {
        self.tcx.local_def_id_to_hir_id(self.item_def_id)
    }

    pub fn node(&self) -> hir::Node<'tcx> {
        self.tcx.hir_node(self.hir_id())
    }

    fn check_tainted_by_errors(&self) -> Result<(), ErrorGuaranteed> {
        match self.tainted_by_errors.get() {
            Some(err) => Err(err),
            None => Ok(()),
        }
    }
}

impl<'tcx> HirTyLowerer<'tcx> for ItemCtxt<'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn dcx(&self) -> DiagCtxtHandle<'_> {
        self.tcx.dcx().taintable_handle(&self.tainted_by_errors)
    }

    fn item_def_id(&self) -> LocalDefId {
        self.item_def_id
    }

    fn re_infer(&self, span: Span, reason: RegionInferReason<'_>) -> ty::Region<'tcx> {
        if let RegionInferReason::ObjectLifetimeDefault = reason {
            let e = struct_span_code_err!(
                self.dcx(),
                span,
                E0228,
                "the lifetime bound for this object type cannot be deduced \
                from context; please supply an explicit bound"
            )
            .emit();
            ty::Region::new_error(self.tcx(), e)
        } else {
            // This indicates an illegal lifetime in a non-assoc-trait position
            ty::Region::new_error_with_message(self.tcx(), span, "unelided lifetime in signature")
        }
    }

    fn ty_infer(&self, _: Option<&ty::GenericParamDef>, span: Span) -> Ty<'tcx> {
        Ty::new_error_with_message(self.tcx(), span, "bad placeholder type")
    }

    fn ct_infer(&self, _: Option<&ty::GenericParamDef>, span: Span) -> Const<'tcx> {
        ty::Const::new_error_with_message(self.tcx(), span, "bad placeholder constant")
    }

    fn probe_ty_param_bounds(
        &self,
        span: Span,
        def_id: LocalDefId,
        assoc_name: Ident,
    ) -> ty::EarlyBinder<'tcx, &'tcx [(ty::Clause<'tcx>, Span)]> {
        self.tcx.at(span).type_param_predicates((self.item_def_id, def_id, assoc_name))
    }

    fn lower_assoc_ty(
        &self,
        span: Span,
        item_def_id: DefId,
        item_segment: &hir::PathSegment<'tcx>,
        poly_trait_ref: ty::PolyTraitRef<'tcx>,
    ) -> Ty<'tcx> {
        if let Some(trait_ref) = poly_trait_ref.no_bound_vars() {
            let item_args = self.lowerer().lower_generic_args_of_assoc_item(
                span,
                item_def_id,
                item_segment,
                trait_ref.args,
            );
            Ty::new_projection_from_args(self.tcx(), item_def_id, item_args)
        } else {
            // There are no late-bound regions; we can just ignore the binder.
            let (mut mpart_sugg, mut inferred_sugg) = (None, None);
            let mut bound = String::new();

            match self.node() {
                hir::Node::Field(_) | hir::Node::Ctor(_) | hir::Node::Variant(_) => {
                    let item = self
                        .tcx
                        .hir()
                        .expect_item(self.tcx.hir().get_parent_item(self.hir_id()).def_id);
                    match &item.kind {
                        hir::ItemKind::Enum(_, generics)
                        | hir::ItemKind::Struct(_, generics)
                        | hir::ItemKind::Union(_, generics) => {
                            let lt_name = get_new_lifetime_name(self.tcx, poly_trait_ref, generics);
                            let (lt_sp, sugg) = match generics.params {
                                [] => (generics.span, format!("<{lt_name}>")),
                                [bound, ..] => (bound.span.shrink_to_lo(), format!("{lt_name}, ")),
                            };
                            mpart_sugg = Some(errors::AssociatedItemTraitUninferredGenericParamsMultipartSuggestion {
                                fspan: lt_sp,
                                first: sugg,
                                sspan: span.with_hi(item_segment.ident.span.lo()),
                                second: format!(
                                    "{}::",
                                    // Replace the existing lifetimes with a new named lifetime.
                                    self.tcx.instantiate_bound_regions_uncached(
                                        poly_trait_ref,
                                        |_| {
                                            ty::Region::new_early_param(self.tcx, ty::EarlyParamRegion {
                                                index: 0,
                                                name: Symbol::intern(&lt_name),
                                            })
                                        }
                                    ),
                                ),
                            });
                        }
                        _ => {}
                    }
                }
                hir::Node::Item(hir::Item {
                    kind:
                        hir::ItemKind::Struct(..) | hir::ItemKind::Enum(..) | hir::ItemKind::Union(..),
                    ..
                }) => {}
                hir::Node::Item(_)
                | hir::Node::ForeignItem(_)
                | hir::Node::TraitItem(_)
                | hir::Node::ImplItem(_) => {
                    inferred_sugg = Some(span.with_hi(item_segment.ident.span.lo()));
                    bound = format!(
                        "{}::",
                        // Erase named lt, we want `<A as B<'_>::C`, not `<A as B<'a>::C`.
                        self.tcx.anonymize_bound_vars(poly_trait_ref).skip_binder(),
                    );
                }
                _ => {}
            }
            Ty::new_error(
                self.tcx(),
                self.tcx().dcx().emit_err(errors::AssociatedItemTraitUninferredGenericParams {
                    span,
                    inferred_sugg,
                    bound,
                    mpart_sugg,
                    what: "type",
                }),
            )
        }
    }

    fn probe_adt(&self, _span: Span, ty: Ty<'tcx>) -> Option<ty::AdtDef<'tcx>> {
        // FIXME(#103640): Should we handle the case where `ty` is a projection?
        ty.ty_adt_def()
    }

    fn record_ty(&self, _hir_id: hir::HirId, _ty: Ty<'tcx>, _span: Span) {
        // There's no place to record types from signatures?
    }

    fn infcx(&self) -> Option<&InferCtxt<'tcx>> {
        None
    }

    fn lower_fn_sig(
        &self,
        decl: &hir::FnDecl<'tcx>,
        generics: Option<&hir::Generics<'_>>,
        hir_id: rustc_hir::HirId,
        hir_ty: Option<&hir::Ty<'_>>,
    ) -> (Vec<Ty<'tcx>>, Ty<'tcx>) {
        let tcx = self.tcx();
        // We proactively collect all the inferred type params to emit a single error per fn def.
        let mut visitor = HirPlaceholderCollector::default();
        let mut infer_replacements = vec![];

        if let Some(generics) = generics {
            walk_generics(&mut visitor, generics);
        }

        let input_tys = decl
            .inputs
            .iter()
            .enumerate()
            .map(|(i, a)| {
                if let hir::TyKind::Infer = a.kind {
                    if let Some(suggested_ty) =
                        self.lowerer().suggest_trait_fn_ty_for_impl_fn_infer(hir_id, Some(i))
                    {
                        infer_replacements.push((a.span, suggested_ty.to_string()));
                        return Ty::new_error_with_message(tcx, a.span, suggested_ty.to_string());
                    }
                }

                // Only visit the type looking for `_` if we didn't fix the type above
                visitor.visit_ty(a);
                self.lowerer().lower_arg_ty(a, None)
            })
            .collect();

        let output_ty = match decl.output {
            hir::FnRetTy::Return(output) => {
                if let hir::TyKind::Infer = output.kind
                    && let Some(suggested_ty) =
                        self.lowerer().suggest_trait_fn_ty_for_impl_fn_infer(hir_id, None)
                {
                    infer_replacements.push((output.span, suggested_ty.to_string()));
                    Ty::new_error_with_message(tcx, output.span, suggested_ty.to_string())
                } else {
                    visitor.visit_ty(output);
                    self.lower_ty(output)
                }
            }
            hir::FnRetTy::DefaultReturn(..) => tcx.types.unit,
        };

        if !(visitor.0.is_empty() && infer_replacements.is_empty()) {
            // We check for the presence of
            // `ident_span` to not emit an error twice when we have `fn foo(_: fn() -> _)`.

            let mut diag = crate::collect::placeholder_type_error_diag(
                self,
                generics,
                visitor.0,
                infer_replacements.iter().map(|(s, _)| *s).collect(),
                true,
                hir_ty,
                "function",
            );

            if !infer_replacements.is_empty() {
                diag.multipart_suggestion(
                    format!(
                    "try replacing `_` with the type{} in the corresponding trait method signature",
                    rustc_errors::pluralize!(infer_replacements.len()),
                ),
                    infer_replacements,
                    Applicability::MachineApplicable,
                );
            }

            diag.emit();
        }

        (input_tys, output_ty)
    }
}

/// Synthesize a new lifetime name that doesn't clash with any of the lifetimes already present.
fn get_new_lifetime_name<'tcx>(
    tcx: TyCtxt<'tcx>,
    poly_trait_ref: ty::PolyTraitRef<'tcx>,
    generics: &hir::Generics<'tcx>,
) -> String {
    let existing_lifetimes = tcx
        .collect_referenced_late_bound_regions(poly_trait_ref)
        .into_iter()
        .filter_map(|lt| {
            if let ty::BoundRegionKind::BrNamed(_, name) = lt {
                Some(name.as_str().to_string())
            } else {
                None
            }
        })
        .chain(generics.params.iter().filter_map(|param| {
            if let hir::GenericParamKind::Lifetime { .. } = &param.kind {
                Some(param.name.ident().as_str().to_string())
            } else {
                None
            }
        }))
        .collect::<FxHashSet<String>>();

    let a_to_z_repeat_n = |n| {
        (b'a'..=b'z').map(move |c| {
            let mut s = '\''.to_string();
            s.extend(std::iter::repeat(char::from(c)).take(n));
            s
        })
    };

    // If all single char lifetime names are present, we wrap around and double the chars.
    (1..).flat_map(a_to_z_repeat_n).find(|lt| !existing_lifetimes.contains(lt.as_str())).unwrap()
}

#[instrument(level = "debug", skip_all)]
fn lower_item(tcx: TyCtxt<'_>, item_id: hir::ItemId) {
    let it = tcx.hir().item(item_id);
    debug!(item = %it.ident, id = %it.hir_id());
    let def_id = item_id.owner_id.def_id;
    let icx = ItemCtxt::new(tcx, def_id);

    match &it.kind {
        // These don't define types.
        hir::ItemKind::ExternCrate(_)
        | hir::ItemKind::Use(..)
        | hir::ItemKind::Macro(..)
        | hir::ItemKind::Mod(_)
        | hir::ItemKind::GlobalAsm(_) => {}
        hir::ItemKind::ForeignMod { items, .. } => {
            for item in *items {
                let item = tcx.hir().foreign_item(item.id);
                tcx.ensure().generics_of(item.owner_id);
                tcx.ensure().type_of(item.owner_id);
                tcx.ensure().predicates_of(item.owner_id);
                match item.kind {
                    hir::ForeignItemKind::Fn(..) => {
                        tcx.ensure().codegen_fn_attrs(item.owner_id);
                        tcx.ensure().fn_sig(item.owner_id)
                    }
                    hir::ForeignItemKind::Static(..) => {
                        tcx.ensure().codegen_fn_attrs(item.owner_id);
                        let mut visitor = HirPlaceholderCollector::default();
                        visitor.visit_foreign_item(item);
                        placeholder_type_error(
                            icx.lowerer(),
                            None,
                            visitor.0,
                            false,
                            None,
                            "static variable",
                        );
                    }
                    _ => (),
                }
            }
        }
        hir::ItemKind::Enum(..) => {
            tcx.ensure().generics_of(def_id);
            tcx.ensure().type_of(def_id);
            tcx.ensure().predicates_of(def_id);
            lower_enum_variant_types(tcx, def_id.to_def_id());
        }
        hir::ItemKind::Impl { .. } => {
            tcx.ensure().generics_of(def_id);
            tcx.ensure().type_of(def_id);
            tcx.ensure().impl_trait_header(def_id);
            tcx.ensure().predicates_of(def_id);
            tcx.ensure().associated_items(def_id);
        }
        hir::ItemKind::Trait(..) => {
            tcx.ensure().generics_of(def_id);
            tcx.ensure().trait_def(def_id);
            tcx.at(it.span).explicit_super_predicates_of(def_id);
            tcx.ensure().predicates_of(def_id);
            tcx.ensure().associated_items(def_id);
        }
        hir::ItemKind::TraitAlias(..) => {
            tcx.ensure().generics_of(def_id);
            tcx.at(it.span).explicit_implied_predicates_of(def_id);
            tcx.at(it.span).explicit_super_predicates_of(def_id);
            tcx.ensure().predicates_of(def_id);
        }
        hir::ItemKind::Struct(struct_def, _) | hir::ItemKind::Union(struct_def, _) => {
            tcx.ensure().generics_of(def_id);
            tcx.ensure().type_of(def_id);
            tcx.ensure().predicates_of(def_id);

            for f in struct_def.fields() {
                tcx.ensure().generics_of(f.def_id);
                tcx.ensure().type_of(f.def_id);
                tcx.ensure().predicates_of(f.def_id);
            }

            if let Some(ctor_def_id) = struct_def.ctor_def_id() {
                lower_variant_ctor(tcx, ctor_def_id);
            }
        }

        hir::ItemKind::TyAlias(..) => {
            tcx.ensure().generics_of(def_id);
            tcx.ensure().type_of(def_id);
            tcx.ensure().predicates_of(def_id);
        }

        hir::ItemKind::Static(ty, ..) | hir::ItemKind::Const(ty, ..) => {
            tcx.ensure().generics_of(def_id);
            tcx.ensure().type_of(def_id);
            tcx.ensure().predicates_of(def_id);
            if !ty.is_suggestable_infer_ty() {
                let mut visitor = HirPlaceholderCollector::default();
                visitor.visit_item(it);
                placeholder_type_error(
                    icx.lowerer(),
                    None,
                    visitor.0,
                    false,
                    None,
                    it.kind.descr(),
                );
            }
        }

        hir::ItemKind::Fn(..) => {
            tcx.ensure().generics_of(def_id);
            tcx.ensure().type_of(def_id);
            tcx.ensure().predicates_of(def_id);
            tcx.ensure().fn_sig(def_id);
            tcx.ensure().codegen_fn_attrs(def_id);
        }
    }
}

fn lower_trait_item(tcx: TyCtxt<'_>, trait_item_id: hir::TraitItemId) {
    let trait_item = tcx.hir().trait_item(trait_item_id);
    let def_id = trait_item_id.owner_id;
    tcx.ensure().generics_of(def_id);
    let icx = ItemCtxt::new(tcx, def_id.def_id);

    match trait_item.kind {
        hir::TraitItemKind::Fn(..) => {
            tcx.ensure().codegen_fn_attrs(def_id);
            tcx.ensure().type_of(def_id);
            tcx.ensure().fn_sig(def_id);
        }

        hir::TraitItemKind::Const(ty, body_id) => {
            tcx.ensure().type_of(def_id);
            if !tcx.dcx().has_stashed_diagnostic(ty.span, StashKey::ItemNoType)
                && !(ty.is_suggestable_infer_ty() && body_id.is_some())
            {
                // Account for `const C: _;`.
                let mut visitor = HirPlaceholderCollector::default();
                visitor.visit_trait_item(trait_item);
                placeholder_type_error(
                    icx.lowerer(),
                    None,
                    visitor.0,
                    false,
                    None,
                    "associated constant",
                );
            }
        }

        hir::TraitItemKind::Type(_, Some(_)) => {
            tcx.ensure().item_bounds(def_id);
            tcx.ensure().item_super_predicates(def_id);
            tcx.ensure().type_of(def_id);
            // Account for `type T = _;`.
            let mut visitor = HirPlaceholderCollector::default();
            visitor.visit_trait_item(trait_item);
            placeholder_type_error(icx.lowerer(), None, visitor.0, false, None, "associated type");
        }

        hir::TraitItemKind::Type(_, None) => {
            tcx.ensure().item_bounds(def_id);
            tcx.ensure().item_super_predicates(def_id);
            // #74612: Visit and try to find bad placeholders
            // even if there is no concrete type.
            let mut visitor = HirPlaceholderCollector::default();
            visitor.visit_trait_item(trait_item);

            placeholder_type_error(icx.lowerer(), None, visitor.0, false, None, "associated type");
        }
    };

    tcx.ensure().predicates_of(def_id);
}

fn lower_impl_item(tcx: TyCtxt<'_>, impl_item_id: hir::ImplItemId) {
    let def_id = impl_item_id.owner_id;
    tcx.ensure().generics_of(def_id);
    tcx.ensure().type_of(def_id);
    tcx.ensure().predicates_of(def_id);
    let impl_item = tcx.hir().impl_item(impl_item_id);
    let icx = ItemCtxt::new(tcx, def_id.def_id);
    match impl_item.kind {
        hir::ImplItemKind::Fn(..) => {
            tcx.ensure().codegen_fn_attrs(def_id);
            tcx.ensure().fn_sig(def_id);
        }
        hir::ImplItemKind::Type(_) => {
            // Account for `type T = _;`
            let mut visitor = HirPlaceholderCollector::default();
            visitor.visit_impl_item(impl_item);

            placeholder_type_error(icx.lowerer(), None, visitor.0, false, None, "associated type");
        }
        hir::ImplItemKind::Const(ty, _) => {
            // Account for `const T: _ = ..;`
            if !ty.is_suggestable_infer_ty() {
                let mut visitor = HirPlaceholderCollector::default();
                visitor.visit_impl_item(impl_item);
                placeholder_type_error(
                    icx.lowerer(),
                    None,
                    visitor.0,
                    false,
                    None,
                    "associated constant",
                );
            }
        }
    }
}

fn lower_variant_ctor(tcx: TyCtxt<'_>, def_id: LocalDefId) {
    tcx.ensure().generics_of(def_id);
    tcx.ensure().type_of(def_id);
    tcx.ensure().predicates_of(def_id);
}

fn lower_enum_variant_types(tcx: TyCtxt<'_>, def_id: DefId) {
    let def = tcx.adt_def(def_id);
    let repr_type = def.repr().discr_type();
    let initial = repr_type.initial_discriminant(tcx);
    let mut prev_discr = None::<Discr<'_>>;

    // fill the discriminant values and field types
    for variant in def.variants() {
        let wrapped_discr = prev_discr.map_or(initial, |d| d.wrap_incr(tcx));
        prev_discr = Some(
            if let ty::VariantDiscr::Explicit(const_def_id) = variant.discr {
                def.eval_explicit_discr(tcx, const_def_id).ok()
            } else if let Some(discr) = repr_type.disr_incr(tcx, prev_discr) {
                Some(discr)
            } else {
                let span = tcx.def_span(variant.def_id);
                tcx.dcx().emit_err(errors::EnumDiscriminantOverflowed {
                    span,
                    discr: prev_discr.unwrap().to_string(),
                    item_name: tcx.item_name(variant.def_id),
                    wrapped_discr: wrapped_discr.to_string(),
                });
                None
            }
            .unwrap_or(wrapped_discr),
        );

        for f in &variant.fields {
            tcx.ensure().generics_of(f.did);
            tcx.ensure().type_of(f.did);
            tcx.ensure().predicates_of(f.did);
        }

        // Lower the ctor, if any. This also registers the variant as an item.
        if let Some(ctor_def_id) = variant.ctor_def_id() {
            lower_variant_ctor(tcx, ctor_def_id.expect_local());
        }
    }
}

#[derive(Clone, Copy)]
struct NestedSpan {
    span: Span,
    nested_field_span: Span,
}

impl NestedSpan {
    fn to_field_already_declared_nested_help(&self) -> errors::FieldAlreadyDeclaredNestedHelp {
        errors::FieldAlreadyDeclaredNestedHelp { span: self.span }
    }
}

#[derive(Clone, Copy)]
enum FieldDeclSpan {
    NotNested(Span),
    Nested(NestedSpan),
}

impl From<Span> for FieldDeclSpan {
    fn from(span: Span) -> Self {
        Self::NotNested(span)
    }
}

impl From<NestedSpan> for FieldDeclSpan {
    fn from(span: NestedSpan) -> Self {
        Self::Nested(span)
    }
}

struct FieldUniquenessCheckContext<'tcx> {
    tcx: TyCtxt<'tcx>,
    seen_fields: FxIndexMap<Ident, FieldDeclSpan>,
}

impl<'tcx> FieldUniquenessCheckContext<'tcx> {
    fn new(tcx: TyCtxt<'tcx>) -> Self {
        Self { tcx, seen_fields: FxIndexMap::default() }
    }

    /// Check if a given field `ident` declared at `field_decl` has been declared elsewhere before.
    fn check_field_decl(&mut self, ident: Ident, field_decl: FieldDeclSpan) {
        use FieldDeclSpan::*;
        let field_name = ident.name;
        let ident = ident.normalize_to_macros_2_0();
        match (field_decl, self.seen_fields.get(&ident).copied()) {
            (NotNested(span), Some(NotNested(prev_span))) => {
                self.tcx.dcx().emit_err(errors::FieldAlreadyDeclared::NotNested {
                    field_name,
                    span,
                    prev_span,
                });
            }
            (NotNested(span), Some(Nested(prev))) => {
                self.tcx.dcx().emit_err(errors::FieldAlreadyDeclared::PreviousNested {
                    field_name,
                    span,
                    prev_span: prev.span,
                    prev_nested_field_span: prev.nested_field_span,
                    prev_help: prev.to_field_already_declared_nested_help(),
                });
            }
            (
                Nested(current @ NestedSpan { span, nested_field_span, .. }),
                Some(NotNested(prev_span)),
            ) => {
                self.tcx.dcx().emit_err(errors::FieldAlreadyDeclared::CurrentNested {
                    field_name,
                    span,
                    nested_field_span,
                    help: current.to_field_already_declared_nested_help(),
                    prev_span,
                });
            }
            (Nested(current @ NestedSpan { span, nested_field_span }), Some(Nested(prev))) => {
                self.tcx.dcx().emit_err(errors::FieldAlreadyDeclared::BothNested {
                    field_name,
                    span,
                    nested_field_span,
                    help: current.to_field_already_declared_nested_help(),
                    prev_span: prev.span,
                    prev_nested_field_span: prev.nested_field_span,
                    prev_help: prev.to_field_already_declared_nested_help(),
                });
            }
            (field_decl, None) => {
                self.seen_fields.insert(ident, field_decl);
            }
        }
    }

    /// Check the uniqueness of fields across adt where there are
    /// nested fields imported from an unnamed field.
    fn check_field_in_nested_adt(&mut self, adt_def: ty::AdtDef<'_>, unnamed_field_span: Span) {
        for field in adt_def.all_fields() {
            if field.is_unnamed() {
                // Here we don't care about the generic parameters, so `instantiate_identity` is enough.
                match self.tcx.type_of(field.did).instantiate_identity().kind() {
                    ty::Adt(adt_def, _) => {
                        self.check_field_in_nested_adt(*adt_def, unnamed_field_span);
                    }
                    ty_kind => span_bug!(
                        self.tcx.def_span(field.did),
                        "Unexpected TyKind in FieldUniquenessCheckContext::check_field_in_nested_adt(): {ty_kind:?}"
                    ),
                }
            } else {
                self.check_field_decl(
                    field.ident(self.tcx),
                    NestedSpan {
                        span: unnamed_field_span,
                        nested_field_span: self.tcx.def_span(field.did),
                    }
                    .into(),
                );
            }
        }
    }

    /// Check the uniqueness of fields in a struct variant, and recursively
    /// check the nested fields if it is an unnamed field with type of an
    /// anonymous adt.
    fn check_field(&mut self, field: &hir::FieldDef<'_>) {
        if field.ident.name != kw::Underscore {
            self.check_field_decl(field.ident, field.span.into());
            return;
        }
        match &field.ty.kind {
            hir::TyKind::AnonAdt(item_id) => {
                match &self.tcx.hir_node(item_id.hir_id()).expect_item().kind {
                    hir::ItemKind::Struct(variant_data, ..)
                    | hir::ItemKind::Union(variant_data, ..) => {
                        variant_data.fields().iter().for_each(|f| self.check_field(f));
                    }
                    item_kind => span_bug!(
                        field.ty.span,
                        "Unexpected ItemKind in FieldUniquenessCheckContext::check_field(): {item_kind:?}"
                    ),
                }
            }
            hir::TyKind::Path(hir::QPath::Resolved(_, hir::Path { res, .. })) => {
                // If this is a direct path to an ADT, we can check it
                // If this is a type alias or non-ADT, `check_unnamed_fields` should verify it
                if let Some(def_id) = res.opt_def_id()
                    && let Some(local) = def_id.as_local()
                    && let Node::Item(item) = self.tcx.hir_node_by_def_id(local)
                    && item.is_adt()
                {
                    self.check_field_in_nested_adt(self.tcx.adt_def(def_id), field.span);
                }
            }
            // Abort due to errors (there must be an error if an unnamed field
            //  has any type kind other than an anonymous adt or a named adt)
            ty_kind => {
                self.tcx.dcx().span_delayed_bug(
                    field.ty.span,
                    format!("Unexpected TyKind in FieldUniquenessCheckContext::check_field(): {ty_kind:?}"),
                );
                // FIXME: errors during AST validation should abort the compilation before reaching here.
                self.tcx.dcx().abort_if_errors();
            }
        }
    }
}

fn lower_variant(
    tcx: TyCtxt<'_>,
    variant_did: Option<LocalDefId>,
    ident: Ident,
    discr: ty::VariantDiscr,
    def: &hir::VariantData<'_>,
    adt_kind: ty::AdtKind,
    parent_did: LocalDefId,
    is_anonymous: bool,
) -> ty::VariantDef {
    let mut has_unnamed_fields = false;
    let mut field_uniqueness_check_ctx = FieldUniquenessCheckContext::new(tcx);
    let fields = def
        .fields()
        .iter()
        .inspect(|f| {
            has_unnamed_fields |= f.ident.name == kw::Underscore;
            // We only check named ADT here because anonymous ADTs are checked inside
            // the named ADT in which they are defined.
            if !is_anonymous {
                field_uniqueness_check_ctx.check_field(f);
            }
        })
        .map(|f| ty::FieldDef {
            did: f.def_id.to_def_id(),
            name: f.ident.name,
            vis: tcx.visibility(f.def_id),
        })
        .collect();
    let recovered = match def {
        hir::VariantData::Struct { recovered: Recovered::Yes(guar), .. } => Some(*guar),
        _ => None,
    };
    ty::VariantDef::new(
        ident.name,
        variant_did.map(LocalDefId::to_def_id),
        def.ctor().map(|(kind, _, def_id)| (kind, def_id.to_def_id())),
        discr,
        fields,
        adt_kind,
        parent_did.to_def_id(),
        recovered,
        adt_kind == AdtKind::Struct && tcx.has_attr(parent_did, sym::non_exhaustive)
            || variant_did
                .is_some_and(|variant_did| tcx.has_attr(variant_did, sym::non_exhaustive)),
        has_unnamed_fields,
    )
}

fn adt_def(tcx: TyCtxt<'_>, def_id: LocalDefId) -> ty::AdtDef<'_> {
    use rustc_hir::*;

    let Node::Item(item) = tcx.hir_node_by_def_id(def_id) else {
        bug!("expected ADT to be an item");
    };

    let is_anonymous = item.ident.name == kw::Empty;
    let repr = if is_anonymous {
        let parent = tcx.local_parent(def_id);
        if let Node::Item(item) = tcx.hir_node_by_def_id(parent)
            && item.is_struct_or_union()
        {
            tcx.adt_def(parent).repr()
        } else {
            tcx.dcx().span_delayed_bug(item.span, "anonymous field inside non struct/union");
            ty::ReprOptions::default()
        }
    } else {
        tcx.repr_options_of_def(def_id)
    };
    let (kind, variants) = match &item.kind {
        ItemKind::Enum(def, _) => {
            let mut distance_from_explicit = 0;
            let variants = def
                .variants
                .iter()
                .map(|v| {
                    let discr = if let Some(e) = &v.disr_expr {
                        distance_from_explicit = 0;
                        ty::VariantDiscr::Explicit(e.def_id.to_def_id())
                    } else {
                        ty::VariantDiscr::Relative(distance_from_explicit)
                    };
                    distance_from_explicit += 1;

                    lower_variant(
                        tcx,
                        Some(v.def_id),
                        v.ident,
                        discr,
                        &v.data,
                        AdtKind::Enum,
                        def_id,
                        is_anonymous,
                    )
                })
                .collect();

            (AdtKind::Enum, variants)
        }
        ItemKind::Struct(def, _) | ItemKind::Union(def, _) => {
            let adt_kind = match item.kind {
                ItemKind::Struct(..) => AdtKind::Struct,
                _ => AdtKind::Union,
            };
            let variants = std::iter::once(lower_variant(
                tcx,
                None,
                item.ident,
                ty::VariantDiscr::Relative(0),
                def,
                adt_kind,
                def_id,
                is_anonymous,
            ))
            .collect();

            (adt_kind, variants)
        }
        _ => bug!("{:?} is not an ADT", item.owner_id.def_id),
    };
    tcx.mk_adt_def(def_id.to_def_id(), kind, variants, repr, is_anonymous)
}

fn trait_def(tcx: TyCtxt<'_>, def_id: LocalDefId) -> ty::TraitDef {
    let item = tcx.hir().expect_item(def_id);

    let (is_alias, is_auto, safety, items) = match item.kind {
        hir::ItemKind::Trait(is_auto, safety, .., items) => {
            (false, is_auto == hir::IsAuto::Yes, safety, items)
        }
        hir::ItemKind::TraitAlias(..) => (true, false, hir::Safety::Safe, &[][..]),
        _ => span_bug!(item.span, "trait_def_of_item invoked on non-trait"),
    };

    // Only regular traits can be const.
    let constness = if !is_alias && tcx.has_attr(def_id, sym::const_trait) {
        hir::Constness::Const
    } else {
        hir::Constness::NotConst
    };

    let paren_sugar = tcx.has_attr(def_id, sym::rustc_paren_sugar);
    if paren_sugar && !tcx.features().unboxed_closures {
        tcx.dcx().emit_err(errors::ParenSugarAttribute { span: item.span });
    }

    // Only regular traits can be marker.
    let is_marker = !is_alias && tcx.has_attr(def_id, sym::marker);

    let rustc_coinductive = tcx.has_attr(def_id, sym::rustc_coinductive);
    let is_fundamental = tcx.has_attr(def_id, sym::fundamental);

    // FIXME: We could probably do way better attribute validation here.
    let mut skip_array_during_method_dispatch = false;
    let mut skip_boxed_slice_during_method_dispatch = false;
    for attr in tcx.get_attrs(def_id, sym::rustc_skip_during_method_dispatch) {
        if let Some(lst) = attr.meta_item_list() {
            for item in lst {
                if let Some(ident) = item.ident() {
                    match ident.as_str() {
                        "array" => skip_array_during_method_dispatch = true,
                        "boxed_slice" => skip_boxed_slice_during_method_dispatch = true,
                        _ => (),
                    }
                }
            }
        }
    }

    let specialization_kind = if tcx.has_attr(def_id, sym::rustc_unsafe_specialization_marker) {
        ty::trait_def::TraitSpecializationKind::Marker
    } else if tcx.has_attr(def_id, sym::rustc_specialization_trait) {
        ty::trait_def::TraitSpecializationKind::AlwaysApplicable
    } else {
        ty::trait_def::TraitSpecializationKind::None
    };
    let must_implement_one_of = tcx
        .get_attr(def_id, sym::rustc_must_implement_one_of)
        // Check that there are at least 2 arguments of `#[rustc_must_implement_one_of]`
        // and that they are all identifiers
        .and_then(|attr| match attr.meta_item_list() {
            Some(items) if items.len() < 2 => {
                tcx.dcx().emit_err(errors::MustImplementOneOfAttribute { span: attr.span });

                None
            }
            Some(items) => items
                .into_iter()
                .map(|item| item.ident().ok_or(item.span()))
                .collect::<Result<Box<[_]>, _>>()
                .map_err(|span| {
                    tcx.dcx().emit_err(errors::MustBeNameOfAssociatedFunction { span });
                })
                .ok()
                .zip(Some(attr.span)),
            // Error is reported by `rustc_attr!`
            None => None,
        })
        // Check that all arguments of `#[rustc_must_implement_one_of]` reference
        // functions in the trait with default implementations
        .and_then(|(list, attr_span)| {
            let errors = list.iter().filter_map(|ident| {
                let item = items.iter().find(|item| item.ident == *ident);

                match item {
                    Some(item) if matches!(item.kind, hir::AssocItemKind::Fn { .. }) => {
                        if !tcx.defaultness(item.id.owner_id).has_value() {
                            tcx.dcx().emit_err(errors::FunctionNotHaveDefaultImplementation {
                                span: item.span,
                                note_span: attr_span,
                            });

                            return Some(());
                        }

                        return None;
                    }
                    Some(item) => {
                        tcx.dcx().emit_err(errors::MustImplementNotFunction {
                            span: item.span,
                            span_note: errors::MustImplementNotFunctionSpanNote { span: attr_span },
                            note: errors::MustImplementNotFunctionNote {},
                        });
                    }
                    None => {
                        tcx.dcx().emit_err(errors::FunctionNotFoundInTrait { span: ident.span });
                    }
                }

                Some(())
            });

            (errors.count() == 0).then_some(list)
        })
        // Check for duplicates
        .and_then(|list| {
            let mut set: UnordMap<Symbol, Span> = Default::default();
            let mut no_dups = true;

            for ident in &*list {
                if let Some(dup) = set.insert(ident.name, ident.span) {
                    tcx.dcx()
                        .emit_err(errors::FunctionNamesDuplicated { spans: vec![dup, ident.span] });

                    no_dups = false;
                }
            }

            no_dups.then_some(list)
        });

    let mut deny_explicit_impl = false;
    let mut implement_via_object = true;
    if let Some(attr) = tcx.get_attr(def_id, sym::rustc_deny_explicit_impl) {
        deny_explicit_impl = true;
        let mut seen_attr = false;
        for meta in attr.meta_item_list().iter().flatten() {
            if let Some(meta) = meta.meta_item()
                && meta.name_or_empty() == sym::implement_via_object
                && let Some(lit) = meta.name_value_literal()
            {
                if seen_attr {
                    tcx.dcx().span_err(meta.span, "duplicated `implement_via_object` meta item");
                }
                seen_attr = true;

                match lit.symbol {
                    kw::True => {
                        implement_via_object = true;
                    }
                    kw::False => {
                        implement_via_object = false;
                    }
                    _ => {
                        tcx.dcx().span_err(
                            meta.span,
                            format!(
                                "unknown literal passed to `implement_via_object` attribute: {}",
                                lit.symbol
                            ),
                        );
                    }
                }
            } else {
                tcx.dcx().span_err(
                    meta.span(),
                    format!("unknown meta item passed to `rustc_deny_explicit_impl` {meta:?}"),
                );
            }
        }
        if !seen_attr {
            tcx.dcx().span_err(attr.span, "missing `implement_via_object` meta item");
        }
    }

    ty::TraitDef {
        def_id: def_id.to_def_id(),
        safety,
        constness,
        paren_sugar,
        has_auto_impl: is_auto,
        is_marker,
        is_coinductive: rustc_coinductive || is_auto,
        is_fundamental,
        skip_array_during_method_dispatch,
        skip_boxed_slice_during_method_dispatch,
        specialization_kind,
        must_implement_one_of,
        implement_via_object,
        deny_explicit_impl,
    }
}

#[instrument(level = "debug", skip(tcx))]
fn fn_sig(tcx: TyCtxt<'_>, def_id: LocalDefId) -> ty::EarlyBinder<'_, ty::PolyFnSig<'_>> {
    use rustc_hir::Node::*;
    use rustc_hir::*;

    let hir_id = tcx.local_def_id_to_hir_id(def_id);

    let icx = ItemCtxt::new(tcx, def_id);

    let output = match tcx.hir_node(hir_id) {
        TraitItem(hir::TraitItem {
            kind: TraitItemKind::Fn(sig, TraitFn::Provided(_)),
            generics,
            ..
        })
        | Item(hir::Item { kind: ItemKind::Fn(sig, generics, _), .. }) => {
            infer_return_ty_for_fn_sig(sig, generics, def_id, &icx)
        }

        ImplItem(hir::ImplItem { kind: ImplItemKind::Fn(sig, _), generics, .. }) => {
            // Do not try to infer the return type for a impl method coming from a trait
            if let Item(hir::Item { kind: ItemKind::Impl(i), .. }) = tcx.parent_hir_node(hir_id)
                && i.of_trait.is_some()
            {
                icx.lowerer().lower_fn_ty(
                    hir_id,
                    sig.header.safety,
                    sig.header.abi,
                    sig.decl,
                    Some(generics),
                    None,
                )
            } else {
                infer_return_ty_for_fn_sig(sig, generics, def_id, &icx)
            }
        }

        TraitItem(hir::TraitItem {
            kind: TraitItemKind::Fn(FnSig { header, decl, span: _ }, _),
            generics,
            ..
        }) => {
            icx.lowerer().lower_fn_ty(hir_id, header.safety, header.abi, decl, Some(generics), None)
        }

        ForeignItem(&hir::ForeignItem { kind: ForeignItemKind::Fn(sig, _, _), .. }) => {
            let abi = tcx.hir().get_foreign_abi(hir_id);
            compute_sig_of_foreign_fn_decl(tcx, def_id, sig.decl, abi, sig.header.safety)
        }

        Ctor(data) | Variant(hir::Variant { data, .. }) if data.ctor().is_some() => {
            let adt_def_id = tcx.hir().get_parent_item(hir_id).def_id.to_def_id();
            let ty = tcx.type_of(adt_def_id).instantiate_identity();
            let inputs = data.fields().iter().map(|f| tcx.type_of(f.def_id).instantiate_identity());
            // constructors for structs with `layout_scalar_valid_range` are unsafe to call
            let safety = match tcx.layout_scalar_valid_range(adt_def_id) {
                (Bound::Unbounded, Bound::Unbounded) => hir::Safety::Safe,
                _ => hir::Safety::Unsafe,
            };
            ty::Binder::dummy(tcx.mk_fn_sig(inputs, ty, false, safety, abi::Abi::Rust))
        }

        Expr(&hir::Expr { kind: hir::ExprKind::Closure { .. }, .. }) => {
            // Closure signatures are not like other function
            // signatures and cannot be accessed through `fn_sig`. For
            // example, a closure signature excludes the `self`
            // argument. In any case they are embedded within the
            // closure type as part of the `ClosureArgs`.
            //
            // To get the signature of a closure, you should use the
            // `sig` method on the `ClosureArgs`:
            //
            //    args.as_closure().sig(def_id, tcx)
            bug!("to get the signature of a closure, use `args.as_closure().sig()` not `fn_sig()`",);
        }

        x => {
            bug!("unexpected sort of node in fn_sig(): {:?}", x);
        }
    };
    ty::EarlyBinder::bind(output)
}

fn infer_return_ty_for_fn_sig<'tcx>(
    sig: &hir::FnSig<'tcx>,
    generics: &hir::Generics<'_>,
    def_id: LocalDefId,
    icx: &ItemCtxt<'tcx>,
) -> ty::PolyFnSig<'tcx> {
    let tcx = icx.tcx;
    let hir_id = tcx.local_def_id_to_hir_id(def_id);

    match sig.decl.output.get_infer_ret_ty() {
        Some(ty) => {
            let fn_sig = tcx.typeck(def_id).liberated_fn_sigs()[hir_id];
            // Typeck doesn't expect erased regions to be returned from `type_of`.
            // This is a heuristic approach. If the scope has region parameters,
            // we should change fn_sig's lifetime from `ReErased` to `ReError`,
            // otherwise to `ReStatic`.
            let has_region_params = generics.params.iter().any(|param| match param.kind {
                GenericParamKind::Lifetime { .. } => true,
                _ => false,
            });
            let fn_sig = tcx.fold_regions(fn_sig, |r, _| match *r {
                ty::ReErased => {
                    if has_region_params {
                        ty::Region::new_error_with_message(
                            tcx,
                            DUMMY_SP,
                            "erased region is not allowed here in return type",
                        )
                    } else {
                        tcx.lifetimes.re_static
                    }
                }
                _ => r,
            });

            let mut visitor = HirPlaceholderCollector::default();
            visitor.visit_ty(ty);

            let mut diag = bad_placeholder(icx.lowerer(), visitor.0, "return type");
            let ret_ty = fn_sig.output();
            // Don't leak types into signatures unless they're nameable!
            // For example, if a function returns itself, we don't want that
            // recursive function definition to leak out into the fn sig.
            let mut recovered_ret_ty = None;

            if let Some(suggestable_ret_ty) = ret_ty.make_suggestable(tcx, false, None) {
                diag.span_suggestion(
                    ty.span,
                    "replace with the correct return type",
                    suggestable_ret_ty,
                    Applicability::MachineApplicable,
                );
                recovered_ret_ty = Some(suggestable_ret_ty);
            } else if let Some(sugg) =
                suggest_impl_trait(&tcx.infer_ctxt().build(), tcx.param_env(def_id), ret_ty)
            {
                diag.span_suggestion(
                    ty.span,
                    "replace with an appropriate return type",
                    sugg,
                    Applicability::MachineApplicable,
                );
            } else if ret_ty.is_closure() {
                diag.help("consider using an `Fn`, `FnMut`, or `FnOnce` trait bound");
            }
            // Also note how `Fn` traits work just in case!
            if ret_ty.is_closure() {
                diag.note(
                    "for more information on `Fn` traits and closure types, see \
                     https://doc.rust-lang.org/book/ch13-01-closures.html",
                );
            }

            let guar = diag.emit();
            ty::Binder::dummy(tcx.mk_fn_sig(
                fn_sig.inputs().iter().copied(),
                recovered_ret_ty.unwrap_or_else(|| Ty::new_error(tcx, guar)),
                fn_sig.c_variadic,
                fn_sig.safety,
                fn_sig.abi,
            ))
        }
        None => icx.lowerer().lower_fn_ty(
            hir_id,
            sig.header.safety,
            sig.header.abi,
            sig.decl,
            Some(generics),
            None,
        ),
    }
}

pub fn suggest_impl_trait<'tcx>(
    infcx: &InferCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    ret_ty: Ty<'tcx>,
) -> Option<String> {
    let format_as_assoc: fn(_, _, _, _, _) -> _ =
        |tcx: TyCtxt<'tcx>,
         _: ty::GenericArgsRef<'tcx>,
         trait_def_id: DefId,
         assoc_item_def_id: DefId,
         item_ty: Ty<'tcx>| {
            let trait_name = tcx.item_name(trait_def_id);
            let assoc_name = tcx.item_name(assoc_item_def_id);
            Some(format!("impl {trait_name}<{assoc_name} = {item_ty}>"))
        };
    let format_as_parenthesized: fn(_, _, _, _, _) -> _ =
        |tcx: TyCtxt<'tcx>,
         args: ty::GenericArgsRef<'tcx>,
         trait_def_id: DefId,
         _: DefId,
         item_ty: Ty<'tcx>| {
            let trait_name = tcx.item_name(trait_def_id);
            let args_tuple = args.type_at(1);
            let ty::Tuple(types) = *args_tuple.kind() else {
                return None;
            };
            let types = types.make_suggestable(tcx, false, None)?;
            let maybe_ret =
                if item_ty.is_unit() { String::new() } else { format!(" -> {item_ty}") };
            Some(format!(
                "impl {trait_name}({}){maybe_ret}",
                types.iter().map(|ty| ty.to_string()).collect::<Vec<_>>().join(", ")
            ))
        };

    for (trait_def_id, assoc_item_def_id, formatter) in [
        (
            infcx.tcx.get_diagnostic_item(sym::Iterator),
            infcx.tcx.get_diagnostic_item(sym::IteratorItem),
            format_as_assoc,
        ),
        (
            infcx.tcx.lang_items().future_trait(),
            infcx.tcx.lang_items().future_output(),
            format_as_assoc,
        ),
        (
            infcx.tcx.lang_items().fn_trait(),
            infcx.tcx.lang_items().fn_once_output(),
            format_as_parenthesized,
        ),
        (
            infcx.tcx.lang_items().fn_mut_trait(),
            infcx.tcx.lang_items().fn_once_output(),
            format_as_parenthesized,
        ),
        (
            infcx.tcx.lang_items().fn_once_trait(),
            infcx.tcx.lang_items().fn_once_output(),
            format_as_parenthesized,
        ),
    ] {
        let Some(trait_def_id) = trait_def_id else {
            continue;
        };
        let Some(assoc_item_def_id) = assoc_item_def_id else {
            continue;
        };
        if infcx.tcx.def_kind(assoc_item_def_id) != DefKind::AssocTy {
            continue;
        }
        let sugg = infcx.probe(|_| {
            let args = ty::GenericArgs::for_item(infcx.tcx, trait_def_id, |param, _| {
                if param.index == 0 { ret_ty.into() } else { infcx.var_for_def(DUMMY_SP, param) }
            });
            if !infcx
                .type_implements_trait(trait_def_id, args, param_env)
                .must_apply_modulo_regions()
            {
                return None;
            }
            let ocx = ObligationCtxt::new(&infcx);
            let item_ty = ocx.normalize(
                &ObligationCause::dummy(),
                param_env,
                Ty::new_projection_from_args(infcx.tcx, assoc_item_def_id, args),
            );
            // FIXME(compiler-errors): We may benefit from resolving regions here.
            if ocx.select_where_possible().is_empty()
                && let item_ty = infcx.resolve_vars_if_possible(item_ty)
                && let Some(item_ty) = item_ty.make_suggestable(infcx.tcx, false, None)
                && let Some(sugg) = formatter(
                    infcx.tcx,
                    infcx.resolve_vars_if_possible(args),
                    trait_def_id,
                    assoc_item_def_id,
                    item_ty,
                )
            {
                return Some(sugg);
            }

            None
        });

        if sugg.is_some() {
            return sugg;
        }
    }
    None
}

fn impl_trait_header(tcx: TyCtxt<'_>, def_id: LocalDefId) -> Option<ty::ImplTraitHeader<'_>> {
    let icx = ItemCtxt::new(tcx, def_id);
    let item = tcx.hir().expect_item(def_id);
    let impl_ = item.expect_impl();
    impl_.of_trait.as_ref().map(|ast_trait_ref| {
        let selfty = tcx.type_of(def_id).instantiate_identity();

        check_impl_constness(tcx, tcx.is_const_trait_impl_raw(def_id.to_def_id()), ast_trait_ref);

        let trait_ref = icx.lowerer().lower_impl_trait_ref(ast_trait_ref, selfty);

        ty::ImplTraitHeader {
            trait_ref: ty::EarlyBinder::bind(trait_ref),
            safety: impl_.safety,
            polarity: polarity_of_impl(tcx, def_id, impl_, item.span),
        }
    })
}

fn check_impl_constness(
    tcx: TyCtxt<'_>,
    is_const: bool,
    hir_trait_ref: &hir::TraitRef<'_>,
) -> Option<ErrorGuaranteed> {
    if !is_const {
        return None;
    }

    let trait_def_id = hir_trait_ref.trait_def_id()?;
    if tcx.is_const_trait(trait_def_id) {
        return None;
    }

    let trait_name = tcx.item_name(trait_def_id).to_string();
    Some(tcx.dcx().emit_err(errors::ConstImplForNonConstTrait {
        trait_ref_span: hir_trait_ref.path.span,
        trait_name,
        local_trait_span:
            trait_def_id.as_local().map(|_| tcx.def_span(trait_def_id).shrink_to_lo()),
        marking: (),
        adding: (),
    }))
}

fn polarity_of_impl(
    tcx: TyCtxt<'_>,
    def_id: LocalDefId,
    impl_: &hir::Impl<'_>,
    span: Span,
) -> ty::ImplPolarity {
    let is_rustc_reservation = tcx.has_attr(def_id, sym::rustc_reservation_impl);
    match &impl_ {
        hir::Impl { polarity: hir::ImplPolarity::Negative(span), of_trait, .. } => {
            if is_rustc_reservation {
                let span = span.to(of_trait.as_ref().map_or(*span, |t| t.path.span));
                tcx.dcx().span_err(span, "reservation impls can't be negative");
            }
            ty::ImplPolarity::Negative
        }
        hir::Impl { polarity: hir::ImplPolarity::Positive, of_trait: None, .. } => {
            if is_rustc_reservation {
                tcx.dcx().span_err(span, "reservation impls can't be inherent");
            }
            ty::ImplPolarity::Positive
        }
        hir::Impl { polarity: hir::ImplPolarity::Positive, of_trait: Some(_), .. } => {
            if is_rustc_reservation {
                ty::ImplPolarity::Reservation
            } else {
                ty::ImplPolarity::Positive
            }
        }
    }
}

/// Returns the early-bound lifetimes declared in this generics
/// listing. For anything other than fns/methods, this is just all
/// the lifetimes that are declared. For fns or methods, we have to
/// screen out those that do not appear in any where-clauses etc using
/// `resolve_lifetime::early_bound_lifetimes`.
fn early_bound_lifetimes_from_generics<'a, 'tcx: 'a>(
    tcx: TyCtxt<'tcx>,
    generics: &'a hir::Generics<'a>,
) -> impl Iterator<Item = &'a hir::GenericParam<'a>> + Captures<'tcx> {
    generics.params.iter().filter(move |param| match param.kind {
        GenericParamKind::Lifetime { .. } => !tcx.is_late_bound(param.hir_id),
        _ => false,
    })
}

fn compute_sig_of_foreign_fn_decl<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: LocalDefId,
    decl: &'tcx hir::FnDecl<'tcx>,
    abi: abi::Abi,
    safety: hir::Safety,
) -> ty::PolyFnSig<'tcx> {
    let safety = if abi == abi::Abi::RustIntrinsic {
        intrinsic_operation_unsafety(tcx, def_id)
    } else {
        safety
    };
    let hir_id = tcx.local_def_id_to_hir_id(def_id);
    let fty =
        ItemCtxt::new(tcx, def_id).lowerer().lower_fn_ty(hir_id, safety, abi, decl, None, None);

    // Feature gate SIMD types in FFI, since I am not sure that the
    // ABIs are handled at all correctly. -huonw
    if abi != abi::Abi::RustIntrinsic && !tcx.features().simd_ffi {
        let check = |hir_ty: &hir::Ty<'_>, ty: Ty<'_>| {
            if ty.is_simd() {
                let snip = tcx
                    .sess
                    .source_map()
                    .span_to_snippet(hir_ty.span)
                    .map_or_else(|_| String::new(), |s| format!(" `{s}`"));
                tcx.dcx().emit_err(errors::SIMDFFIHighlyExperimental { span: hir_ty.span, snip });
            }
        };
        for (input, ty) in iter::zip(decl.inputs, fty.inputs().skip_binder()) {
            check(input, *ty)
        }
        if let hir::FnRetTy::Return(ty) = decl.output {
            check(ty, fty.output().skip_binder())
        }
    }

    fty
}

fn coroutine_kind(tcx: TyCtxt<'_>, def_id: LocalDefId) -> Option<hir::CoroutineKind> {
    match tcx.hir_node_by_def_id(def_id) {
        Node::Expr(&hir::Expr {
            kind:
                hir::ExprKind::Closure(&rustc_hir::Closure {
                    kind: hir::ClosureKind::Coroutine(kind),
                    ..
                }),
            ..
        }) => Some(kind),
        _ => None,
    }
}

fn coroutine_for_closure(tcx: TyCtxt<'_>, def_id: LocalDefId) -> DefId {
    let &rustc_hir::Closure { kind: hir::ClosureKind::CoroutineClosure(_), body, .. } =
        tcx.hir_node_by_def_id(def_id).expect_closure()
    else {
        bug!()
    };

    let &hir::Expr {
        kind:
            hir::ExprKind::Closure(&rustc_hir::Closure {
                def_id,
                kind: hir::ClosureKind::Coroutine(_),
                ..
            }),
        ..
    } = tcx.hir().body(body).value
    else {
        bug!()
    };

    def_id.to_def_id()
}

fn is_type_alias_impl_trait<'tcx>(tcx: TyCtxt<'tcx>, def_id: LocalDefId) -> bool {
    let opaque = tcx.hir().expect_opaque_ty(def_id);
    matches!(opaque.origin, hir::OpaqueTyOrigin::TyAlias { .. })
}

fn rendered_precise_capturing_args<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: LocalDefId,
) -> Option<&'tcx [Symbol]> {
    if let Some(ty::ImplTraitInTraitData::Trait { opaque_def_id, .. }) =
        tcx.opt_rpitit_info(def_id.to_def_id())
    {
        return tcx.rendered_precise_capturing_args(opaque_def_id);
    }

    tcx.hir_node_by_def_id(def_id).expect_opaque_ty().bounds.iter().find_map(|bound| match bound {
        hir::GenericBound::Use(args, ..) => {
            Some(&*tcx.arena.alloc_from_iter(args.iter().map(|arg| arg.name())))
        }
        _ => None,
    })
}
