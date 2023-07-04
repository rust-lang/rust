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

use crate::astconv::AstConv;
use crate::check::intrinsic::intrinsic_operation_unsafety;
use crate::errors;
use hir::def::DefKind;
use rustc_data_structures::captures::Captures;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_errors::{Applicability, DiagnosticBuilder, ErrorGuaranteed, StashKey};
use rustc_hir as hir;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_hir::intravisit::{self, Visitor};
use rustc_hir::{GenericParamKind, Node};
use rustc_infer::infer::{InferCtxt, TyCtxtInferExt};
use rustc_infer::traits::ObligationCause;
use rustc_middle::hir::nested_filter;
use rustc_middle::query::Providers;
use rustc_middle::ty::util::{Discr, IntTypeExt};
use rustc_middle::ty::{self, AdtKind, Const, IsSuggestable, ToPredicate, Ty, TyCtxt};
use rustc_span::symbol::{kw, sym, Ident, Symbol};
use rustc_span::Span;
use rustc_target::spec::abi;
use rustc_trait_selection::infer::InferCtxtExt;
use rustc_trait_selection::traits::error_reporting::suggestions::NextTypeParamName;
use rustc_trait_selection::traits::ObligationCtxt;
use std::iter;

mod generics_of;
mod item_bounds;
mod predicates_of;
mod resolve_bound_vars;
mod type_of;

///////////////////////////////////////////////////////////////////////////
// Main entry point

fn collect_mod_item_types(tcx: TyCtxt<'_>, module_def_id: LocalDefId) {
    tcx.hir().visit_item_likes_in_module(module_def_id, &mut CollectItemTypesVisitor { tcx });
}

pub fn provide(providers: &mut Providers) {
    resolve_bound_vars::provide(providers);
    *providers = Providers {
        type_of: type_of::type_of,
        item_bounds: item_bounds::item_bounds,
        explicit_item_bounds: item_bounds::explicit_item_bounds,
        generics_of: generics_of::generics_of,
        predicates_of: predicates_of::predicates_of,
        predicates_defined_on,
        explicit_predicates_of: predicates_of::explicit_predicates_of,
        super_predicates_of: predicates_of::super_predicates_of,
        implied_predicates_of: predicates_of::implied_predicates_of,
        super_predicates_that_define_assoc_item:
            predicates_of::super_predicates_that_define_assoc_item,
        trait_explicit_predicates_and_bounds: predicates_of::trait_explicit_predicates_and_bounds,
        type_param_predicates: predicates_of::type_param_predicates,
        trait_def,
        adt_def,
        fn_sig,
        impl_trait_ref,
        impl_polarity,
        generator_kind,
        collect_mod_item_types,
        is_type_alias_impl_trait,
        ..*providers
    };
}

///////////////////////////////////////////////////////////////////////////

/// Context specific to some particular item. This is what implements
/// [`AstConv`].
///
/// # `ItemCtxt` vs `FnCtxt`
///
/// `ItemCtxt` is primarily used to type-check item signatures and lower them
/// from HIR to their [`ty::Ty`] representation, which is exposed using [`AstConv`].
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
/// process. So we can't just store a pointer to e.g., the AST or the
/// parsed ty form, we have to be more flexible. To this end, the
/// `ItemCtxt` is parameterized by a `DefId` that it uses to satisfy
/// `get_type_parameter_bounds` requests, drawing the information from
/// the AST (`hir::Generics`), recursively.
pub struct ItemCtxt<'tcx> {
    tcx: TyCtxt<'tcx>,
    item_def_id: LocalDefId,
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
    fn visit_array_length(&mut self, length: &'v hir::ArrayLen) {
        if let &hir::ArrayLen::Infer(_, span) = length {
            self.0.push(span);
        }
        intravisit::walk_array_len(self, length)
    }
}

struct CollectItemTypesVisitor<'tcx> {
    tcx: TyCtxt<'tcx>,
}

/// If there are any placeholder types (`_`), emit an error explaining that this is not allowed
/// and suggest adding type parameters in the appropriate place, taking into consideration any and
/// all already existing generic type parameters to avoid suggesting a name that is already in use.
pub(crate) fn placeholder_type_error<'tcx>(
    tcx: TyCtxt<'tcx>,
    generics: Option<&hir::Generics<'_>>,
    placeholder_types: Vec<Span>,
    suggest: bool,
    hir_ty: Option<&hir::Ty<'_>>,
    kind: &'static str,
) {
    if placeholder_types.is_empty() {
        return;
    }

    placeholder_type_error_diag(tcx, generics, placeholder_types, vec![], suggest, hir_ty, kind)
        .emit();
}

pub(crate) fn placeholder_type_error_diag<'tcx>(
    tcx: TyCtxt<'tcx>,
    generics: Option<&hir::Generics<'_>>,
    placeholder_types: Vec<Span>,
    additional_spans: Vec<Span>,
    suggest: bool,
    hir_ty: Option<&hir::Ty<'_>>,
    kind: &'static str,
) -> DiagnosticBuilder<'tcx, ErrorGuaranteed> {
    if placeholder_types.is_empty() {
        return bad_placeholder(tcx, additional_spans, kind);
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
            sugg.push((span, format!(", {}", type_name)));
        } else {
            sugg.push((generics.span, format!("<{}>", type_name)));
        }
    }

    let mut err =
        bad_placeholder(tcx, placeholder_types.into_iter().chain(additional_spans).collect(), kind);

    // Suggest, but only if it is not a function in const or static
    if suggest {
        let mut is_fn = false;
        let mut is_const_or_static = false;

        if let Some(hir_ty) = hir_ty && let hir::TyKind::BareFn(_) = hir_ty.kind {
            is_fn = true;

            // Check if parent is const or static
            let parent_id = tcx.hir().parent_id(hir_ty.hir_id);
            let parent_node = tcx.hir().get(parent_id);

            is_const_or_static = matches!(
                parent_node,
                Node::Item(&hir::Item {
                    kind: hir::ItemKind::Const(..) | hir::ItemKind::Static(..),
                    ..
                }) | Node::TraitItem(&hir::TraitItem {
                    kind: hir::TraitItemKind::Const(..),
                    ..
                }) | Node::ImplItem(&hir::ImplItem { kind: hir::ImplItemKind::Const(..), .. })
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
        hir::ItemKind::OpaqueTy(hir::OpaqueTy { generics, .. })
        | hir::ItemKind::TyAlias(_, generics) => (generics, false),
        // `static`, `fn` and `const` are handled elsewhere to suggest appropriate type.
        _ => return,
    };

    let mut visitor = HirPlaceholderCollector::default();
    visitor.visit_item(item);

    placeholder_type_error(tcx, Some(generics), visitor.0, suggest, None, item.kind.descr());
}

impl<'tcx> Visitor<'tcx> for CollectItemTypesVisitor<'tcx> {
    type NestedFilter = nested_filter::OnlyBodies;

    fn nested_visit_map(&mut self) -> Self::Map {
        self.tcx.hir()
    }

    fn visit_item(&mut self, item: &'tcx hir::Item<'tcx>) {
        convert_item(self.tcx, item.item_id());
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
                        self.tcx.ensure().type_of(default.def_id);
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

    fn visit_trait_item(&mut self, trait_item: &'tcx hir::TraitItem<'tcx>) {
        convert_trait_item(self.tcx, trait_item.trait_item_id());
        intravisit::walk_trait_item(self, trait_item);
    }

    fn visit_impl_item(&mut self, impl_item: &'tcx hir::ImplItem<'tcx>) {
        convert_impl_item(self.tcx, impl_item.impl_item_id());
        intravisit::walk_impl_item(self, impl_item);
    }
}

///////////////////////////////////////////////////////////////////////////
// Utility types and common code for the above passes.

fn bad_placeholder<'tcx>(
    tcx: TyCtxt<'tcx>,
    mut spans: Vec<Span>,
    kind: &'static str,
) -> DiagnosticBuilder<'tcx, ErrorGuaranteed> {
    let kind = if kind.ends_with('s') { format!("{}es", kind) } else { format!("{}s", kind) };

    spans.sort();
    tcx.sess.create_err(errors::PlaceholderNotAllowedItemSignatures { spans, kind })
}

impl<'tcx> ItemCtxt<'tcx> {
    pub fn new(tcx: TyCtxt<'tcx>, item_def_id: LocalDefId) -> ItemCtxt<'tcx> {
        ItemCtxt { tcx, item_def_id }
    }

    pub fn to_ty(&self, ast_ty: &hir::Ty<'_>) -> Ty<'tcx> {
        self.astconv().ast_ty_to_ty(ast_ty)
    }

    pub fn hir_id(&self) -> hir::HirId {
        self.tcx.hir().local_def_id_to_hir_id(self.item_def_id)
    }

    pub fn node(&self) -> hir::Node<'tcx> {
        self.tcx.hir().get(self.hir_id())
    }
}

impl<'tcx> AstConv<'tcx> for ItemCtxt<'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn item_def_id(&self) -> DefId {
        self.item_def_id.to_def_id()
    }

    fn get_type_parameter_bounds(
        &self,
        span: Span,
        def_id: LocalDefId,
        assoc_name: Ident,
    ) -> ty::GenericPredicates<'tcx> {
        self.tcx.at(span).type_param_predicates((self.item_def_id, def_id, assoc_name))
    }

    fn re_infer(&self, _: Option<&ty::GenericParamDef>, _: Span) -> Option<ty::Region<'tcx>> {
        None
    }

    fn allow_ty_infer(&self) -> bool {
        false
    }

    fn ty_infer(&self, _: Option<&ty::GenericParamDef>, span: Span) -> Ty<'tcx> {
        self.tcx().ty_error_with_message(span, "bad placeholder type")
    }

    fn ct_infer(&self, ty: Ty<'tcx>, _: Option<&ty::GenericParamDef>, span: Span) -> Const<'tcx> {
        let ty = self.tcx.fold_regions(ty, |r, _| match *r {
            // This is never reached in practice. If it ever is reached,
            // `ReErased` should be changed to `ReStatic`, and any other region
            // left alone.
            r => bug!("unexpected region: {r:?}"),
        });
        ty::Const::new_error_with_message(self.tcx(), ty, span, "bad placeholder constant")
    }

    fn projected_ty_from_poly_trait_ref(
        &self,
        span: Span,
        item_def_id: DefId,
        item_segment: &hir::PathSegment<'_>,
        poly_trait_ref: ty::PolyTraitRef<'tcx>,
    ) -> Ty<'tcx> {
        if let Some(trait_ref) = poly_trait_ref.no_bound_vars() {
            let item_substs = self.astconv().create_substs_for_associated_item(
                span,
                item_def_id,
                item_segment,
                trait_ref.substs,
            );
            self.tcx().mk_projection(item_def_id, item_substs)
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
                                [] => (generics.span, format!("<{}>", lt_name)),
                                [bound, ..] => {
                                    (bound.span.shrink_to_lo(), format!("{}, ", lt_name))
                                }
                            };
                            mpart_sugg = Some(errors::AssociatedTypeTraitUninferredGenericParamsMultipartSuggestion {
                                fspan: lt_sp,
                                first: sugg,
                                sspan: span.with_hi(item_segment.ident.span.lo()),
                                second: format!(
                                    "{}::",
                                    // Replace the existing lifetimes with a new named lifetime.
                                    self.tcx.replace_late_bound_regions_uncached(
                                        poly_trait_ref,
                                        |_| {
                                            ty::Region::new_early_bound(self.tcx, ty::EarlyBoundRegion {
                                                def_id: item_def_id,
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
            self.tcx().ty_error(self.tcx().sess.emit_err(
                errors::AssociatedTypeTraitUninferredGenericParams {
                    span,
                    inferred_sugg,
                    bound,
                    mpart_sugg,
                },
            ))
        }
    }

    fn probe_adt(&self, _span: Span, ty: Ty<'tcx>) -> Option<ty::AdtDef<'tcx>> {
        // FIXME(#103640): Should we handle the case where `ty` is a projection?
        ty.ty_adt_def()
    }

    fn set_tainted_by_errors(&self, _: ErrorGuaranteed) {
        // There's no obvious place to track this, so just let it go.
    }

    fn record_ty(&self, _hir_id: hir::HirId, _ty: Ty<'tcx>, _span: Span) {
        // There's no place to record types from signatures?
    }

    fn infcx(&self) -> Option<&InferCtxt<'tcx>> {
        None
    }
}

/// Synthesize a new lifetime name that doesn't clash with any of the lifetimes already present.
fn get_new_lifetime_name<'tcx>(
    tcx: TyCtxt<'tcx>,
    poly_trait_ref: ty::PolyTraitRef<'tcx>,
    generics: &hir::Generics<'tcx>,
) -> String {
    let existing_lifetimes = tcx
        .collect_referenced_late_bound_regions(&poly_trait_ref)
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

fn convert_item(tcx: TyCtxt<'_>, item_id: hir::ItemId) {
    let it = tcx.hir().item(item_id);
    debug!("convert: item {} with id {}", it.ident, it.hir_id());
    let def_id = item_id.owner_id.def_id;

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
                            tcx,
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
            convert_enum_variant_types(tcx, def_id.to_def_id());
        }
        hir::ItemKind::Impl { .. } => {
            tcx.ensure().generics_of(def_id);
            tcx.ensure().type_of(def_id);
            tcx.ensure().impl_trait_ref(def_id);
            tcx.ensure().predicates_of(def_id);
        }
        hir::ItemKind::Trait(..) => {
            tcx.ensure().generics_of(def_id);
            tcx.ensure().trait_def(def_id);
            tcx.at(it.span).super_predicates_of(def_id);
            tcx.ensure().predicates_of(def_id);
        }
        hir::ItemKind::TraitAlias(..) => {
            tcx.ensure().generics_of(def_id);
            tcx.at(it.span).implied_predicates_of(def_id);
            tcx.at(it.span).super_predicates_of(def_id);
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
                convert_variant_ctor(tcx, ctor_def_id);
            }
        }

        // Don't call `type_of` on opaque types, since that depends on type
        // checking function bodies. `check_item_type` ensures that it's called
        // instead.
        hir::ItemKind::OpaqueTy(..) => {
            tcx.ensure().generics_of(def_id);
            tcx.ensure().predicates_of(def_id);
            tcx.ensure().explicit_item_bounds(def_id);
            tcx.ensure().item_bounds(def_id);
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
            if !is_suggestable_infer_ty(ty) {
                let mut visitor = HirPlaceholderCollector::default();
                visitor.visit_item(it);
                placeholder_type_error(tcx, None, visitor.0, false, None, it.kind.descr());
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

fn convert_trait_item(tcx: TyCtxt<'_>, trait_item_id: hir::TraitItemId) {
    let trait_item = tcx.hir().trait_item(trait_item_id);
    let def_id = trait_item_id.owner_id;
    tcx.ensure().generics_of(def_id);

    match trait_item.kind {
        hir::TraitItemKind::Fn(..) => {
            tcx.ensure().codegen_fn_attrs(def_id);
            tcx.ensure().type_of(def_id);
            tcx.ensure().fn_sig(def_id);
        }

        hir::TraitItemKind::Const(ty, body_id) => {
            tcx.ensure().type_of(def_id);
            if !tcx.sess.diagnostic().has_stashed_diagnostic(ty.span, StashKey::ItemNoType)
                && !(is_suggestable_infer_ty(ty) && body_id.is_some())
            {
                // Account for `const C: _;`.
                let mut visitor = HirPlaceholderCollector::default();
                visitor.visit_trait_item(trait_item);
                placeholder_type_error(tcx, None, visitor.0, false, None, "associated constant");
            }
        }

        hir::TraitItemKind::Type(_, Some(_)) => {
            tcx.ensure().item_bounds(def_id);
            tcx.ensure().type_of(def_id);
            // Account for `type T = _;`.
            let mut visitor = HirPlaceholderCollector::default();
            visitor.visit_trait_item(trait_item);
            placeholder_type_error(tcx, None, visitor.0, false, None, "associated type");
        }

        hir::TraitItemKind::Type(_, None) => {
            tcx.ensure().item_bounds(def_id);
            // #74612: Visit and try to find bad placeholders
            // even if there is no concrete type.
            let mut visitor = HirPlaceholderCollector::default();
            visitor.visit_trait_item(trait_item);

            placeholder_type_error(tcx, None, visitor.0, false, None, "associated type");
        }
    };

    tcx.ensure().predicates_of(def_id);
}

fn convert_impl_item(tcx: TyCtxt<'_>, impl_item_id: hir::ImplItemId) {
    let def_id = impl_item_id.owner_id;
    tcx.ensure().generics_of(def_id);
    tcx.ensure().type_of(def_id);
    tcx.ensure().predicates_of(def_id);
    let impl_item = tcx.hir().impl_item(impl_item_id);
    match impl_item.kind {
        hir::ImplItemKind::Fn(..) => {
            tcx.ensure().codegen_fn_attrs(def_id);
            tcx.ensure().fn_sig(def_id);
        }
        hir::ImplItemKind::Type(_) => {
            // Account for `type T = _;`
            let mut visitor = HirPlaceholderCollector::default();
            visitor.visit_impl_item(impl_item);

            placeholder_type_error(tcx, None, visitor.0, false, None, "associated type");
        }
        hir::ImplItemKind::Const(ty, _) => {
            // Account for `const T: _ = ..;`
            if !is_suggestable_infer_ty(ty) {
                let mut visitor = HirPlaceholderCollector::default();
                visitor.visit_impl_item(impl_item);
                placeholder_type_error(tcx, None, visitor.0, false, None, "associated constant");
            }
        }
    }
}

fn convert_variant_ctor(tcx: TyCtxt<'_>, def_id: LocalDefId) {
    tcx.ensure().generics_of(def_id);
    tcx.ensure().type_of(def_id);
    tcx.ensure().predicates_of(def_id);
}

fn convert_enum_variant_types(tcx: TyCtxt<'_>, def_id: DefId) {
    let def = tcx.adt_def(def_id);
    let repr_type = def.repr().discr_type();
    let initial = repr_type.initial_discriminant(tcx);
    let mut prev_discr = None::<Discr<'_>>;

    // fill the discriminant values and field types
    for variant in def.variants() {
        let wrapped_discr = prev_discr.map_or(initial, |d| d.wrap_incr(tcx));
        prev_discr = Some(
            if let ty::VariantDiscr::Explicit(const_def_id) = variant.discr {
                def.eval_explicit_discr(tcx, const_def_id)
            } else if let Some(discr) = repr_type.disr_incr(tcx, prev_discr) {
                Some(discr)
            } else {
                let span = tcx.def_span(variant.def_id);
                tcx.sess.emit_err(errors::EnumDiscriminantOverflowed {
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

        // Convert the ctor, if any. This also registers the variant as
        // an item.
        if let Some(ctor_def_id) = variant.ctor_def_id() {
            convert_variant_ctor(tcx, ctor_def_id.expect_local());
        }
    }
}

fn convert_variant(
    tcx: TyCtxt<'_>,
    variant_did: Option<LocalDefId>,
    ident: Ident,
    discr: ty::VariantDiscr,
    def: &hir::VariantData<'_>,
    adt_kind: ty::AdtKind,
    parent_did: LocalDefId,
) -> ty::VariantDef {
    let mut seen_fields: FxHashMap<Ident, Span> = Default::default();
    let fields = def
        .fields()
        .iter()
        .map(|f| {
            let dup_span = seen_fields.get(&f.ident.normalize_to_macros_2_0()).cloned();
            if let Some(prev_span) = dup_span {
                tcx.sess.emit_err(errors::FieldAlreadyDeclared {
                    field_name: f.ident,
                    span: f.span,
                    prev_span,
                });
            } else {
                seen_fields.insert(f.ident.normalize_to_macros_2_0(), f.span);
            }

            ty::FieldDef {
                did: f.def_id.to_def_id(),
                name: f.ident.name,
                vis: tcx.visibility(f.def_id),
            }
        })
        .collect();
    let recovered = match def {
        hir::VariantData::Struct(_, r) => *r,
        _ => false,
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
    )
}

fn adt_def(tcx: TyCtxt<'_>, def_id: LocalDefId) -> ty::AdtDef<'_> {
    use rustc_hir::*;

    let hir_id = tcx.hir().local_def_id_to_hir_id(def_id);
    let Node::Item(item) = tcx.hir().get(hir_id) else {
        bug!();
    };

    let repr = tcx.repr_options_of_def(def_id.to_def_id());
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

                    convert_variant(
                        tcx,
                        Some(v.def_id),
                        v.ident,
                        discr,
                        &v.data,
                        AdtKind::Enum,
                        def_id,
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
            let variants = std::iter::once(convert_variant(
                tcx,
                None,
                item.ident,
                ty::VariantDiscr::Relative(0),
                def,
                adt_kind,
                def_id,
            ))
            .collect();

            (adt_kind, variants)
        }
        _ => bug!(),
    };
    tcx.mk_adt_def(def_id.to_def_id(), kind, variants, repr)
}

fn trait_def(tcx: TyCtxt<'_>, def_id: LocalDefId) -> ty::TraitDef {
    let item = tcx.hir().expect_item(def_id);

    let (is_auto, unsafety, items) = match item.kind {
        hir::ItemKind::Trait(is_auto, unsafety, .., items) => {
            (is_auto == hir::IsAuto::Yes, unsafety, items)
        }
        hir::ItemKind::TraitAlias(..) => (false, hir::Unsafety::Normal, &[][..]),
        _ => span_bug!(item.span, "trait_def_of_item invoked on non-trait"),
    };

    let paren_sugar = tcx.has_attr(def_id, sym::rustc_paren_sugar);
    if paren_sugar && !tcx.features().unboxed_closures {
        tcx.sess.emit_err(errors::ParenSugarAttribute { span: item.span });
    }

    let is_marker = tcx.has_attr(def_id, sym::marker);
    let rustc_coinductive = tcx.has_attr(def_id, sym::rustc_coinductive);
    let skip_array_during_method_dispatch =
        tcx.has_attr(def_id, sym::rustc_skip_array_during_method_dispatch);
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
                tcx.sess.emit_err(errors::MustImplementOneOfAttribute { span: attr.span });

                None
            }
            Some(items) => items
                .into_iter()
                .map(|item| item.ident().ok_or(item.span()))
                .collect::<Result<Box<[_]>, _>>()
                .map_err(|span| {
                    tcx.sess.emit_err(errors::MustBeNameOfAssociatedFunction { span });
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
                            tcx.sess.emit_err(errors::FunctionNotHaveDefaultImplementation {
                                span: item.span,
                                note_span: attr_span,
                            });

                            return Some(());
                        }

                        return None;
                    }
                    Some(item) => {
                        tcx.sess.emit_err(errors::MustImplementNotFunction {
                            span: item.span,
                            span_note: errors::MustImplementNotFunctionSpanNote { span: attr_span },
                            note: errors::MustImplementNotFunctionNote {},
                        });
                    }
                    None => {
                        tcx.sess.emit_err(errors::FunctionNotFoundInTrait { span: ident.span });
                    }
                }

                Some(())
            });

            (errors.count() == 0).then_some(list)
        })
        // Check for duplicates
        .and_then(|list| {
            let mut set: FxHashMap<Symbol, Span> = FxHashMap::default();
            let mut no_dups = true;

            for ident in &*list {
                if let Some(dup) = set.insert(ident.name, ident.span) {
                    tcx.sess
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
                    tcx.sess.span_err(
                        meta.span,
                        "duplicated `implement_via_object` meta item",
                    );
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
                        tcx.sess.span_err(
                            meta.span,
                            format!("unknown literal passed to `implement_via_object` attribute: {}", lit.symbol),
                        );
                    }
                }
            } else {
                tcx.sess.span_err(
                    meta.span(),
                    format!("unknown meta item passed to `rustc_deny_explicit_impl` {:?}", meta),
                );
            }
        }
        if !seen_attr {
            tcx.sess.span_err(attr.span, "missing `implement_via_object` meta item");
        }
    }

    ty::TraitDef {
        def_id: def_id.to_def_id(),
        unsafety,
        paren_sugar,
        has_auto_impl: is_auto,
        is_marker,
        is_coinductive: rustc_coinductive || is_auto,
        skip_array_during_method_dispatch,
        specialization_kind,
        must_implement_one_of,
        implement_via_object,
        deny_explicit_impl,
    }
}

fn are_suggestable_generic_args(generic_args: &[hir::GenericArg<'_>]) -> bool {
    generic_args.iter().any(|arg| match arg {
        hir::GenericArg::Type(ty) => is_suggestable_infer_ty(ty),
        hir::GenericArg::Infer(_) => true,
        _ => false,
    })
}

/// Whether `ty` is a type with `_` placeholders that can be inferred. Used in diagnostics only to
/// use inference to provide suggestions for the appropriate type if possible.
fn is_suggestable_infer_ty(ty: &hir::Ty<'_>) -> bool {
    debug!(?ty);
    use hir::TyKind::*;
    match &ty.kind {
        Infer => true,
        Slice(ty) => is_suggestable_infer_ty(ty),
        Array(ty, length) => {
            is_suggestable_infer_ty(ty) || matches!(length, hir::ArrayLen::Infer(_, _))
        }
        Tup(tys) => tys.iter().any(is_suggestable_infer_ty),
        Ptr(mut_ty) | Ref(_, mut_ty) => is_suggestable_infer_ty(mut_ty.ty),
        OpaqueDef(_, generic_args, _) => are_suggestable_generic_args(generic_args),
        Path(hir::QPath::TypeRelative(ty, segment)) => {
            is_suggestable_infer_ty(ty) || are_suggestable_generic_args(segment.args().args)
        }
        Path(hir::QPath::Resolved(ty_opt, hir::Path { segments, .. })) => {
            ty_opt.is_some_and(is_suggestable_infer_ty)
                || segments.iter().any(|segment| are_suggestable_generic_args(segment.args().args))
        }
        _ => false,
    }
}

pub fn get_infer_ret_ty<'hir>(output: &'hir hir::FnRetTy<'hir>) -> Option<&'hir hir::Ty<'hir>> {
    if let hir::FnRetTy::Return(ty) = output {
        if is_suggestable_infer_ty(ty) {
            return Some(&*ty);
        }
    }
    None
}

#[instrument(level = "debug", skip(tcx))]
fn fn_sig(tcx: TyCtxt<'_>, def_id: LocalDefId) -> ty::EarlyBinder<ty::PolyFnSig<'_>> {
    use rustc_hir::Node::*;
    use rustc_hir::*;

    let hir_id = tcx.hir().local_def_id_to_hir_id(def_id);

    let icx = ItemCtxt::new(tcx, def_id);

    let output = match tcx.hir().get(hir_id) {
        TraitItem(hir::TraitItem {
            kind: TraitItemKind::Fn(sig, TraitFn::Provided(_)),
            generics,
            ..
        })
        | Item(hir::Item { kind: ItemKind::Fn(sig, generics, _), .. }) => {
            infer_return_ty_for_fn_sig(tcx, sig, generics, def_id, &icx)
        }

        ImplItem(hir::ImplItem { kind: ImplItemKind::Fn(sig, _), generics, .. }) => {
            // Do not try to infer the return type for a impl method coming from a trait
            if let Item(hir::Item { kind: ItemKind::Impl(i), .. }) =
                tcx.hir().get_parent(hir_id)
                && i.of_trait.is_some()
            {
                icx.astconv().ty_of_fn(
                    hir_id,
                    sig.header.unsafety,
                    sig.header.abi,
                    sig.decl,
                    Some(generics),
                    None,
                )
            } else {
                infer_return_ty_for_fn_sig(tcx, sig, generics, def_id, &icx)
            }
        }

        TraitItem(hir::TraitItem {
            kind: TraitItemKind::Fn(FnSig { header, decl, span: _ }, _),
            generics,
            ..
        }) => {
            icx.astconv().ty_of_fn(hir_id, header.unsafety, header.abi, decl, Some(generics), None)
        }

        ForeignItem(&hir::ForeignItem { kind: ForeignItemKind::Fn(fn_decl, _, _), .. }) => {
            let abi = tcx.hir().get_foreign_abi(hir_id);
            compute_sig_of_foreign_fn_decl(tcx, def_id, fn_decl, abi)
        }

        Ctor(data) | Variant(hir::Variant { data, .. }) if data.ctor().is_some() => {
            let ty = tcx.type_of(tcx.hir().get_parent_item(hir_id)).subst_identity();
            let inputs = data.fields().iter().map(|f| tcx.type_of(f.def_id).subst_identity());
            ty::Binder::dummy(tcx.mk_fn_sig(
                inputs,
                ty,
                false,
                hir::Unsafety::Normal,
                abi::Abi::Rust,
            ))
        }

        Expr(&hir::Expr { kind: hir::ExprKind::Closure { .. }, .. }) => {
            // Closure signatures are not like other function
            // signatures and cannot be accessed through `fn_sig`. For
            // example, a closure signature excludes the `self`
            // argument. In any case they are embedded within the
            // closure type as part of the `ClosureSubsts`.
            //
            // To get the signature of a closure, you should use the
            // `sig` method on the `ClosureSubsts`:
            //
            //    substs.as_closure().sig(def_id, tcx)
            bug!(
                "to get the signature of a closure, use `substs.as_closure().sig()` not `fn_sig()`",
            );
        }

        x => {
            bug!("unexpected sort of node in fn_sig(): {:?}", x);
        }
    };
    ty::EarlyBinder::bind(output)
}

fn infer_return_ty_for_fn_sig<'tcx>(
    tcx: TyCtxt<'tcx>,
    sig: &hir::FnSig<'_>,
    generics: &hir::Generics<'_>,
    def_id: LocalDefId,
    icx: &ItemCtxt<'tcx>,
) -> ty::PolyFnSig<'tcx> {
    let hir_id = tcx.hir().local_def_id_to_hir_id(def_id);

    match get_infer_ret_ty(&sig.decl.output) {
        Some(ty) => {
            let fn_sig = tcx.typeck(def_id).liberated_fn_sigs()[hir_id];
            // Typeck doesn't expect erased regions to be returned from `type_of`.
            let fn_sig = tcx.fold_regions(fn_sig, |r, _| match *r {
                ty::ReErased => tcx.lifetimes.re_static,
                _ => r,
            });

            let mut visitor = HirPlaceholderCollector::default();
            visitor.visit_ty(ty);

            let mut diag = bad_placeholder(tcx, visitor.0, "return type");
            let ret_ty = fn_sig.output();
            // Don't leak types into signatures unless they're nameable!
            // For example, if a function returns itself, we don't want that
            // recursive function definition to leak out into the fn sig.
            let mut should_recover = false;

            if let Some(ret_ty) = ret_ty.make_suggestable(tcx, false) {
                diag.span_suggestion(
                    ty.span,
                    "replace with the correct return type",
                    ret_ty,
                    Applicability::MachineApplicable,
                );
                should_recover = true;
            } else if let Some(sugg) = suggest_impl_trait(tcx, ret_ty, ty.span, def_id) {
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

            if should_recover {
                ty::Binder::dummy(fn_sig)
            } else {
                ty::Binder::dummy(tcx.mk_fn_sig(
                    fn_sig.inputs().iter().copied(),
                    tcx.ty_error(guar),
                    fn_sig.c_variadic,
                    fn_sig.unsafety,
                    fn_sig.abi,
                ))
            }
        }
        None => icx.astconv().ty_of_fn(
            hir_id,
            sig.header.unsafety,
            sig.header.abi,
            sig.decl,
            Some(generics),
            None,
        ),
    }
}

fn suggest_impl_trait<'tcx>(
    tcx: TyCtxt<'tcx>,
    ret_ty: Ty<'tcx>,
    span: Span,
    def_id: LocalDefId,
) -> Option<String> {
    let format_as_assoc: fn(_, _, _, _, _) -> _ =
        |tcx: TyCtxt<'tcx>,
         _: ty::SubstsRef<'tcx>,
         trait_def_id: DefId,
         assoc_item_def_id: DefId,
         item_ty: Ty<'tcx>| {
            let trait_name = tcx.item_name(trait_def_id);
            let assoc_name = tcx.item_name(assoc_item_def_id);
            Some(format!("impl {trait_name}<{assoc_name} = {item_ty}>"))
        };
    let format_as_parenthesized: fn(_, _, _, _, _) -> _ =
        |tcx: TyCtxt<'tcx>,
         substs: ty::SubstsRef<'tcx>,
         trait_def_id: DefId,
         _: DefId,
         item_ty: Ty<'tcx>| {
            let trait_name = tcx.item_name(trait_def_id);
            let args_tuple = substs.type_at(1);
            let ty::Tuple(types) = *args_tuple.kind() else { return None; };
            let types = types.make_suggestable(tcx, false)?;
            let maybe_ret =
                if item_ty.is_unit() { String::new() } else { format!(" -> {item_ty}") };
            Some(format!(
                "impl {trait_name}({}){maybe_ret}",
                types.iter().map(|ty| ty.to_string()).collect::<Vec<_>>().join(", ")
            ))
        };

    for (trait_def_id, assoc_item_def_id, formatter) in [
        (
            tcx.get_diagnostic_item(sym::Iterator),
            tcx.get_diagnostic_item(sym::IteratorItem),
            format_as_assoc,
        ),
        (
            tcx.lang_items().future_trait(),
            tcx.get_diagnostic_item(sym::FutureOutput),
            format_as_assoc,
        ),
        (tcx.lang_items().fn_trait(), tcx.lang_items().fn_once_output(), format_as_parenthesized),
        (
            tcx.lang_items().fn_mut_trait(),
            tcx.lang_items().fn_once_output(),
            format_as_parenthesized,
        ),
        (
            tcx.lang_items().fn_once_trait(),
            tcx.lang_items().fn_once_output(),
            format_as_parenthesized,
        ),
    ] {
        let Some(trait_def_id) = trait_def_id else { continue; };
        let Some(assoc_item_def_id) = assoc_item_def_id else { continue; };
        if tcx.def_kind(assoc_item_def_id) != DefKind::AssocTy {
            continue;
        }
        let param_env = tcx.param_env(def_id);
        let infcx = tcx.infer_ctxt().build();
        let substs = ty::InternalSubsts::for_item(tcx, trait_def_id, |param, _| {
            if param.index == 0 { ret_ty.into() } else { infcx.var_for_def(span, param) }
        });
        if !infcx.type_implements_trait(trait_def_id, substs, param_env).must_apply_modulo_regions()
        {
            continue;
        }
        let ocx = ObligationCtxt::new(&infcx);
        let item_ty = ocx.normalize(
            &ObligationCause::misc(span, def_id),
            param_env,
            tcx.mk_projection(assoc_item_def_id, substs),
        );
        // FIXME(compiler-errors): We may benefit from resolving regions here.
        if ocx.select_where_possible().is_empty()
            && let item_ty = infcx.resolve_vars_if_possible(item_ty)
            && let Some(item_ty) = item_ty.make_suggestable(tcx, false)
            && let Some(sugg) = formatter(tcx, infcx.resolve_vars_if_possible(substs), trait_def_id, assoc_item_def_id, item_ty)
        {
            return Some(sugg);
        }
    }
    None
}

fn impl_trait_ref(
    tcx: TyCtxt<'_>,
    def_id: LocalDefId,
) -> Option<ty::EarlyBinder<ty::TraitRef<'_>>> {
    let icx = ItemCtxt::new(tcx, def_id);
    let impl_ = tcx.hir().expect_item(def_id).expect_impl();
    impl_
        .of_trait
        .as_ref()
        .map(|ast_trait_ref| {
            let selfty = tcx.type_of(def_id).subst_identity();
            icx.astconv().instantiate_mono_trait_ref(
                ast_trait_ref,
                selfty,
                check_impl_constness(tcx, impl_.constness, ast_trait_ref),
            )
        })
        .map(ty::EarlyBinder::bind)
}

fn check_impl_constness(
    tcx: TyCtxt<'_>,
    constness: hir::Constness,
    ast_trait_ref: &hir::TraitRef<'_>,
) -> ty::BoundConstness {
    match constness {
        hir::Constness::Const => {
            if let Some(trait_def_id) = ast_trait_ref.trait_def_id() && !tcx.has_attr(trait_def_id, sym::const_trait) {
                let trait_name = tcx.item_name(trait_def_id).to_string();
                tcx.sess.emit_err(errors::ConstImplForNonConstTrait {
                    trait_ref_span: ast_trait_ref.path.span,
                    trait_name,
                    local_trait_span: trait_def_id.as_local().map(|_| tcx.def_span(trait_def_id).shrink_to_lo()),
                    marking: (),
                    adding: (),
                });
                ty::BoundConstness::NotConst
            } else {
                ty::BoundConstness::ConstIfConst
            }
        },
        hir::Constness::NotConst => ty::BoundConstness::NotConst,
    }
}

fn impl_polarity(tcx: TyCtxt<'_>, def_id: LocalDefId) -> ty::ImplPolarity {
    let is_rustc_reservation = tcx.has_attr(def_id, sym::rustc_reservation_impl);
    let item = tcx.hir().expect_item(def_id);
    match &item.kind {
        hir::ItemKind::Impl(hir::Impl {
            polarity: hir::ImplPolarity::Negative(span),
            of_trait,
            ..
        }) => {
            if is_rustc_reservation {
                let span = span.to(of_trait.as_ref().map_or(*span, |t| t.path.span));
                tcx.sess.span_err(span, "reservation impls can't be negative");
            }
            ty::ImplPolarity::Negative
        }
        hir::ItemKind::Impl(hir::Impl {
            polarity: hir::ImplPolarity::Positive,
            of_trait: None,
            ..
        }) => {
            if is_rustc_reservation {
                tcx.sess.span_err(item.span, "reservation impls can't be inherent");
            }
            ty::ImplPolarity::Positive
        }
        hir::ItemKind::Impl(hir::Impl {
            polarity: hir::ImplPolarity::Positive,
            of_trait: Some(_),
            ..
        }) => {
            if is_rustc_reservation {
                ty::ImplPolarity::Reservation
            } else {
                ty::ImplPolarity::Positive
            }
        }
        item => bug!("impl_polarity: {:?} not an impl", item),
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

/// Returns a list of type predicates for the definition with ID `def_id`, including inferred
/// lifetime constraints. This includes all predicates returned by `explicit_predicates_of`, plus
/// inferred constraints concerning which regions outlive other regions.
#[instrument(level = "debug", skip(tcx))]
fn predicates_defined_on(tcx: TyCtxt<'_>, def_id: DefId) -> ty::GenericPredicates<'_> {
    let mut result = tcx.explicit_predicates_of(def_id);
    debug!("predicates_defined_on: explicit_predicates_of({:?}) = {:?}", def_id, result,);
    let inferred_outlives = tcx.inferred_outlives_of(def_id);
    if !inferred_outlives.is_empty() {
        debug!(
            "predicates_defined_on: inferred_outlives_of({:?}) = {:?}",
            def_id, inferred_outlives,
        );
        let inferred_outlives_iter =
            inferred_outlives.iter().map(|(clause, span)| ((*clause).to_predicate(tcx), *span));
        if result.predicates.is_empty() {
            result.predicates = tcx.arena.alloc_from_iter(inferred_outlives_iter);
        } else {
            result.predicates = tcx.arena.alloc_from_iter(
                result.predicates.into_iter().copied().chain(inferred_outlives_iter),
            );
        }
    }

    debug!("predicates_defined_on({:?}) = {:?}", def_id, result);
    result
}

fn compute_sig_of_foreign_fn_decl<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: LocalDefId,
    decl: &'tcx hir::FnDecl<'tcx>,
    abi: abi::Abi,
) -> ty::PolyFnSig<'tcx> {
    let unsafety = if abi == abi::Abi::RustIntrinsic {
        intrinsic_operation_unsafety(tcx, def_id.to_def_id())
    } else {
        hir::Unsafety::Unsafe
    };
    let hir_id = tcx.hir().local_def_id_to_hir_id(def_id);
    let fty =
        ItemCtxt::new(tcx, def_id).astconv().ty_of_fn(hir_id, unsafety, abi, decl, None, None);

    // Feature gate SIMD types in FFI, since I am not sure that the
    // ABIs are handled at all correctly. -huonw
    if abi != abi::Abi::RustIntrinsic
        && abi != abi::Abi::PlatformIntrinsic
        && !tcx.features().simd_ffi
    {
        let check = |ast_ty: &hir::Ty<'_>, ty: Ty<'_>| {
            if ty.is_simd() {
                let snip = tcx
                    .sess
                    .source_map()
                    .span_to_snippet(ast_ty.span)
                    .map_or_else(|_| String::new(), |s| format!(" `{}`", s));
                tcx.sess.emit_err(errors::SIMDFFIHighlyExperimental { span: ast_ty.span, snip });
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

fn generator_kind(tcx: TyCtxt<'_>, def_id: LocalDefId) -> Option<hir::GeneratorKind> {
    match tcx.hir().get_by_def_id(def_id) {
        Node::Expr(&rustc_hir::Expr {
            kind: rustc_hir::ExprKind::Closure(&rustc_hir::Closure { body, .. }),
            ..
        }) => tcx.hir().body(body).generator_kind(),
        _ => None,
    }
}

fn is_type_alias_impl_trait<'tcx>(tcx: TyCtxt<'tcx>, def_id: LocalDefId) -> bool {
    match tcx.hir().get_by_def_id(def_id) {
        Node::Item(hir::Item { kind: hir::ItemKind::OpaqueTy(opaque), .. }) => {
            matches!(opaque.origin, hir::OpaqueTyOrigin::TyAlias { .. })
        }
        _ => bug!("tried getting opaque_ty_origin for non-opaque: {:?}", def_id),
    }
}
