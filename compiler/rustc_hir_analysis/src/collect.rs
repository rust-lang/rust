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
use rustc_ast as ast;
use rustc_ast::{MetaItemKind, NestedMetaItem};
use rustc_attr::{list_contains_name, InlineAttr, InstructionSetAttr, OptimizeAttr};
use rustc_data_structures::captures::Captures;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_errors::{struct_span_err, Applicability, DiagnosticBuilder, ErrorGuaranteed, StashKey};
use rustc_hir as hir;
use rustc_hir::def::CtorKind;
use rustc_hir::def_id::{DefId, LocalDefId, LOCAL_CRATE};
use rustc_hir::intravisit::{self, Visitor};
use rustc_hir::weak_lang_items;
use rustc_hir::{GenericParamKind, Node};
use rustc_middle::hir::nested_filter;
use rustc_middle::middle::codegen_fn_attrs::{CodegenFnAttrFlags, CodegenFnAttrs};
use rustc_middle::mir::mono::Linkage;
use rustc_middle::ty::query::Providers;
use rustc_middle::ty::util::{Discr, IntTypeExt};
use rustc_middle::ty::ReprOptions;
use rustc_middle::ty::{self, AdtKind, Const, DefIdTree, IsSuggestable, Ty, TyCtxt};
use rustc_session::lint;
use rustc_session::parse::feature_err;
use rustc_span::symbol::{kw, sym, Ident, Symbol};
use rustc_span::Span;
use rustc_target::spec::{abi, SanitizerSet};
use rustc_trait_selection::traits::error_reporting::suggestions::NextTypeParamName;
use std::iter;

mod generics_of;
mod item_bounds;
mod predicates_of;
mod type_of;

///////////////////////////////////////////////////////////////////////////
// Main entry point

fn collect_mod_item_types(tcx: TyCtxt<'_>, module_def_id: LocalDefId) {
    tcx.hir().visit_item_likes_in_module(module_def_id, &mut CollectItemTypesVisitor { tcx });
}

pub fn provide(providers: &mut Providers) {
    *providers = Providers {
        opt_const_param_of: type_of::opt_const_param_of,
        type_of: type_of::type_of,
        item_bounds: item_bounds::item_bounds,
        explicit_item_bounds: item_bounds::explicit_item_bounds,
        generics_of: generics_of::generics_of,
        predicates_of: predicates_of::predicates_of,
        predicates_defined_on,
        explicit_predicates_of: predicates_of::explicit_predicates_of,
        super_predicates_of: predicates_of::super_predicates_of,
        super_predicates_that_define_assoc_type:
            predicates_of::super_predicates_that_define_assoc_type,
        trait_explicit_predicates_and_bounds: predicates_of::trait_explicit_predicates_and_bounds,
        type_param_predicates: predicates_of::type_param_predicates,
        trait_def,
        adt_def,
        fn_sig,
        impl_trait_ref,
        impl_polarity,
        is_foreign_item,
        generator_kind,
        codegen_fn_attrs,
        asm_target_features,
        collect_mod_item_types,
        should_inherit_track_caller,
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
/// This is in contrast to [`FnCtxt`], which is used to type-check bodies of
/// functions, closures, and `const`s -- anywhere that expressions and statements show up.
///
/// An important thing to note is that `ItemCtxt` does no inference -- it has no [`InferCtxt`] --
/// while `FnCtxt` does do inference.
///
/// [`FnCtxt`]: crate::check::FnCtxt
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
    item_def_id: DefId,
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
            let parent_id = tcx.hir().get_parent_node(hir_ty.hir_id);
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
                    let def_id = self.tcx.hir().local_def_id(param.hir_id);
                    self.tcx.ensure().type_of(def_id);
                }
                hir::GenericParamKind::Type { .. } => {}
                hir::GenericParamKind::Const { default, .. } => {
                    let def_id = self.tcx.hir().local_def_id(param.hir_id);
                    self.tcx.ensure().type_of(def_id);
                    if let Some(default) = default {
                        let default_def_id = self.tcx.hir().local_def_id(default.hir_id);
                        // need to store default and type of default
                        self.tcx.ensure().type_of(default_def_id);
                        self.tcx.ensure().const_param_default(def_id);
                    }
                }
            }
        }
        intravisit::walk_generics(self, generics);
    }

    fn visit_expr(&mut self, expr: &'tcx hir::Expr<'tcx>) {
        if let hir::ExprKind::Closure { .. } = expr.kind {
            let def_id = self.tcx.hir().local_def_id(expr.hir_id);
            self.tcx.ensure().generics_of(def_id);
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
    let mut err = struct_span_err!(
        tcx.sess,
        spans.clone(),
        E0121,
        "the placeholder `_` is not allowed within types on item signatures for {}",
        kind
    );
    for span in spans {
        err.span_label(span, "not allowed in type signatures");
    }
    err
}

impl<'tcx> ItemCtxt<'tcx> {
    pub fn new(tcx: TyCtxt<'tcx>, item_def_id: DefId) -> ItemCtxt<'tcx> {
        ItemCtxt { tcx, item_def_id }
    }

    pub fn to_ty(&self, ast_ty: &hir::Ty<'_>) -> Ty<'tcx> {
        <dyn AstConv<'_>>::ast_ty_to_ty(self, ast_ty)
    }

    pub fn hir_id(&self) -> hir::HirId {
        self.tcx.hir().local_def_id_to_hir_id(self.item_def_id.expect_local())
    }

    pub fn node(&self) -> hir::Node<'tcx> {
        self.tcx.hir().get(self.hir_id())
    }
}

impl<'tcx> AstConv<'tcx> for ItemCtxt<'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn item_def_id(&self) -> Option<DefId> {
        Some(self.item_def_id)
    }

    fn get_type_parameter_bounds(
        &self,
        span: Span,
        def_id: DefId,
        assoc_name: Ident,
    ) -> ty::GenericPredicates<'tcx> {
        self.tcx.at(span).type_param_predicates((
            self.item_def_id,
            def_id.expect_local(),
            assoc_name,
        ))
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
            ty::ReErased => self.tcx.lifetimes.re_static,
            _ => r,
        });
        self.tcx().const_error_with_message(ty, span, "bad placeholder constant")
    }

    fn projected_ty_from_poly_trait_ref(
        &self,
        span: Span,
        item_def_id: DefId,
        item_segment: &hir::PathSegment<'_>,
        poly_trait_ref: ty::PolyTraitRef<'tcx>,
    ) -> Ty<'tcx> {
        if let Some(trait_ref) = poly_trait_ref.no_bound_vars() {
            let item_substs = <dyn AstConv<'tcx>>::create_substs_for_associated_item(
                self,
                span,
                item_def_id,
                item_segment,
                trait_ref.substs,
            );
            self.tcx().mk_projection(item_def_id, item_substs)
        } else {
            // There are no late-bound regions; we can just ignore the binder.
            let mut err = struct_span_err!(
                self.tcx().sess,
                span,
                E0212,
                "cannot use the associated type of a trait \
                 with uninferred generic parameters"
            );

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
                            let suggestions = vec![
                                (lt_sp, sugg),
                                (
                                    span.with_hi(item_segment.ident.span.lo()),
                                    format!(
                                        "{}::",
                                        // Replace the existing lifetimes with a new named lifetime.
                                        self.tcx.replace_late_bound_regions_uncached(
                                            poly_trait_ref,
                                            |_| {
                                                self.tcx.mk_region(ty::ReEarlyBound(
                                                    ty::EarlyBoundRegion {
                                                        def_id: item_def_id,
                                                        index: 0,
                                                        name: Symbol::intern(&lt_name),
                                                    },
                                                ))
                                            }
                                        ),
                                    ),
                                ),
                            ];
                            err.multipart_suggestion(
                                "use a fully qualified path with explicit lifetimes",
                                suggestions,
                                Applicability::MaybeIncorrect,
                            );
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
                    err.span_suggestion_verbose(
                        span.with_hi(item_segment.ident.span.lo()),
                        "use a fully qualified path with inferred lifetimes",
                        format!(
                            "{}::",
                            // Erase named lt, we want `<A as B<'_>::C`, not `<A as B<'a>::C`.
                            self.tcx.anonymize_late_bound_regions(poly_trait_ref).skip_binder(),
                        ),
                        Applicability::MaybeIncorrect,
                    );
                }
                _ => {}
            }
            err.emit();
            self.tcx().ty_error()
        }
    }

    fn normalize_ty(&self, _span: Span, ty: Ty<'tcx>) -> Ty<'tcx> {
        // Types in item signatures are not normalized to avoid undue dependencies.
        ty
    }

    fn set_tainted_by_errors(&self) {
        // There's no obvious place to track this, so just let it go.
    }

    fn record_ty(&self, _hir_id: hir::HirId, _ty: Ty<'tcx>, _span: Span) {
        // There's no place to record types from signatures?
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
    let def_id = item_id.def_id.def_id;

    match it.kind {
        // These don't define types.
        hir::ItemKind::ExternCrate(_)
        | hir::ItemKind::Use(..)
        | hir::ItemKind::Macro(..)
        | hir::ItemKind::Mod(_)
        | hir::ItemKind::GlobalAsm(_) => {}
        hir::ItemKind::ForeignMod { items, .. } => {
            for item in items {
                let item = tcx.hir().foreign_item(item.id);
                tcx.ensure().generics_of(item.def_id);
                tcx.ensure().type_of(item.def_id);
                tcx.ensure().predicates_of(item.def_id);
                match item.kind {
                    hir::ForeignItemKind::Fn(..) => tcx.ensure().fn_sig(item.def_id),
                    hir::ForeignItemKind::Static(..) => {
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
        hir::ItemKind::Enum(ref enum_definition, _) => {
            tcx.ensure().generics_of(def_id);
            tcx.ensure().type_of(def_id);
            tcx.ensure().predicates_of(def_id);
            convert_enum_variant_types(tcx, def_id.to_def_id(), enum_definition.variants);
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
            tcx.at(it.span).super_predicates_of(def_id);
            tcx.ensure().predicates_of(def_id);
        }
        hir::ItemKind::Struct(ref struct_def, _) | hir::ItemKind::Union(ref struct_def, _) => {
            tcx.ensure().generics_of(def_id);
            tcx.ensure().type_of(def_id);
            tcx.ensure().predicates_of(def_id);

            for f in struct_def.fields() {
                let def_id = tcx.hir().local_def_id(f.hir_id);
                tcx.ensure().generics_of(def_id);
                tcx.ensure().type_of(def_id);
                tcx.ensure().predicates_of(def_id);
            }

            if let Some(ctor_hir_id) = struct_def.ctor_hir_id() {
                convert_variant_ctor(tcx, ctor_hir_id);
            }
        }

        // Desugared from `impl Trait`, so visited by the function's return type.
        hir::ItemKind::OpaqueTy(hir::OpaqueTy {
            origin: hir::OpaqueTyOrigin::FnReturn(..) | hir::OpaqueTyOrigin::AsyncFn(..),
            ..
        }) => {}

        // Don't call `type_of` on opaque types, since that depends on type
        // checking function bodies. `check_item_type` ensures that it's called
        // instead.
        hir::ItemKind::OpaqueTy(..) => {
            tcx.ensure().generics_of(def_id);
            tcx.ensure().predicates_of(def_id);
            tcx.ensure().explicit_item_bounds(def_id);
        }
        hir::ItemKind::TyAlias(..)
        | hir::ItemKind::Static(..)
        | hir::ItemKind::Const(..)
        | hir::ItemKind::Fn(..) => {
            tcx.ensure().generics_of(def_id);
            tcx.ensure().type_of(def_id);
            tcx.ensure().predicates_of(def_id);
            match it.kind {
                hir::ItemKind::Fn(..) => tcx.ensure().fn_sig(def_id),
                hir::ItemKind::OpaqueTy(..) => tcx.ensure().item_bounds(def_id),
                hir::ItemKind::Const(ty, ..) | hir::ItemKind::Static(ty, ..) => {
                    if !is_suggestable_infer_ty(ty) {
                        let mut visitor = HirPlaceholderCollector::default();
                        visitor.visit_item(it);
                        placeholder_type_error(tcx, None, visitor.0, false, None, it.kind.descr());
                    }
                }
                _ => (),
            }
        }
    }
}

fn convert_trait_item(tcx: TyCtxt<'_>, trait_item_id: hir::TraitItemId) {
    let trait_item = tcx.hir().trait_item(trait_item_id);
    let def_id = trait_item_id.def_id;
    tcx.ensure().generics_of(def_id);

    match trait_item.kind {
        hir::TraitItemKind::Fn(..) => {
            tcx.ensure().type_of(def_id);
            tcx.ensure().fn_sig(def_id);
        }

        hir::TraitItemKind::Const(.., Some(_)) => {
            tcx.ensure().type_of(def_id);
        }

        hir::TraitItemKind::Const(hir_ty, _) => {
            tcx.ensure().type_of(def_id);
            // Account for `const C: _;`.
            let mut visitor = HirPlaceholderCollector::default();
            visitor.visit_trait_item(trait_item);
            if !tcx.sess.diagnostic().has_stashed_diagnostic(hir_ty.span, StashKey::ItemNoType) {
                placeholder_type_error(tcx, None, visitor.0, false, None, "constant");
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
    let def_id = impl_item_id.def_id;
    tcx.ensure().generics_of(def_id);
    tcx.ensure().type_of(def_id);
    tcx.ensure().predicates_of(def_id);
    let impl_item = tcx.hir().impl_item(impl_item_id);
    match impl_item.kind {
        hir::ImplItemKind::Fn(..) => {
            tcx.ensure().fn_sig(def_id);
        }
        hir::ImplItemKind::TyAlias(_) => {
            // Account for `type T = _;`
            let mut visitor = HirPlaceholderCollector::default();
            visitor.visit_impl_item(impl_item);

            placeholder_type_error(tcx, None, visitor.0, false, None, "associated type");
        }
        hir::ImplItemKind::Const(..) => {}
    }
}

fn convert_variant_ctor(tcx: TyCtxt<'_>, ctor_id: hir::HirId) {
    let def_id = tcx.hir().local_def_id(ctor_id);
    tcx.ensure().generics_of(def_id);
    tcx.ensure().type_of(def_id);
    tcx.ensure().predicates_of(def_id);
}

fn convert_enum_variant_types(tcx: TyCtxt<'_>, def_id: DefId, variants: &[hir::Variant<'_>]) {
    let def = tcx.adt_def(def_id);
    let repr_type = def.repr().discr_type();
    let initial = repr_type.initial_discriminant(tcx);
    let mut prev_discr = None::<Discr<'_>>;

    // fill the discriminant values and field types
    for variant in variants {
        let wrapped_discr = prev_discr.map_or(initial, |d| d.wrap_incr(tcx));
        prev_discr = Some(
            if let Some(ref e) = variant.disr_expr {
                let expr_did = tcx.hir().local_def_id(e.hir_id);
                def.eval_explicit_discr(tcx, expr_did.to_def_id())
            } else if let Some(discr) = repr_type.disr_incr(tcx, prev_discr) {
                Some(discr)
            } else {
                struct_span_err!(tcx.sess, variant.span, E0370, "enum discriminant overflowed")
                    .span_label(
                        variant.span,
                        format!("overflowed on value after {}", prev_discr.unwrap()),
                    )
                    .note(&format!(
                        "explicitly set `{} = {}` if that is desired outcome",
                        variant.ident, wrapped_discr
                    ))
                    .emit();
                None
            }
            .unwrap_or(wrapped_discr),
        );

        for f in variant.data.fields() {
            let def_id = tcx.hir().local_def_id(f.hir_id);
            tcx.ensure().generics_of(def_id);
            tcx.ensure().type_of(def_id);
            tcx.ensure().predicates_of(def_id);
        }

        // Convert the ctor, if any. This also registers the variant as
        // an item.
        if let Some(ctor_hir_id) = variant.data.ctor_hir_id() {
            convert_variant_ctor(tcx, ctor_hir_id);
        }
    }
}

fn convert_variant(
    tcx: TyCtxt<'_>,
    variant_did: Option<LocalDefId>,
    ctor_did: Option<LocalDefId>,
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
            let fid = tcx.hir().local_def_id(f.hir_id);
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

            ty::FieldDef { did: fid.to_def_id(), name: f.ident.name, vis: tcx.visibility(fid) }
        })
        .collect();
    let recovered = match def {
        hir::VariantData::Struct(_, r) => *r,
        _ => false,
    };
    ty::VariantDef::new(
        ident.name,
        variant_did.map(LocalDefId::to_def_id),
        ctor_did.map(LocalDefId::to_def_id),
        discr,
        fields,
        CtorKind::from_hir(def),
        adt_kind,
        parent_did.to_def_id(),
        recovered,
        adt_kind == AdtKind::Struct && tcx.has_attr(parent_did.to_def_id(), sym::non_exhaustive)
            || variant_did.map_or(false, |variant_did| {
                tcx.has_attr(variant_did.to_def_id(), sym::non_exhaustive)
            }),
    )
}

fn adt_def<'tcx>(tcx: TyCtxt<'tcx>, def_id: DefId) -> ty::AdtDef<'tcx> {
    use rustc_hir::*;

    let def_id = def_id.expect_local();
    let hir_id = tcx.hir().local_def_id_to_hir_id(def_id);
    let Node::Item(item) = tcx.hir().get(hir_id) else {
        bug!();
    };

    let repr = ReprOptions::new(tcx, def_id.to_def_id());
    let (kind, variants) = match item.kind {
        ItemKind::Enum(ref def, _) => {
            let mut distance_from_explicit = 0;
            let variants = def
                .variants
                .iter()
                .map(|v| {
                    let variant_did = Some(tcx.hir().local_def_id(v.id));
                    let ctor_did =
                        v.data.ctor_hir_id().map(|hir_id| tcx.hir().local_def_id(hir_id));

                    let discr = if let Some(ref e) = v.disr_expr {
                        distance_from_explicit = 0;
                        ty::VariantDiscr::Explicit(tcx.hir().local_def_id(e.hir_id).to_def_id())
                    } else {
                        ty::VariantDiscr::Relative(distance_from_explicit)
                    };
                    distance_from_explicit += 1;

                    convert_variant(
                        tcx,
                        variant_did,
                        ctor_did,
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
        ItemKind::Struct(ref def, _) => {
            let variant_did = None::<LocalDefId>;
            let ctor_did = def.ctor_hir_id().map(|hir_id| tcx.hir().local_def_id(hir_id));

            let variants = std::iter::once(convert_variant(
                tcx,
                variant_did,
                ctor_did,
                item.ident,
                ty::VariantDiscr::Relative(0),
                def,
                AdtKind::Struct,
                def_id,
            ))
            .collect();

            (AdtKind::Struct, variants)
        }
        ItemKind::Union(ref def, _) => {
            let variant_did = None;
            let ctor_did = def.ctor_hir_id().map(|hir_id| tcx.hir().local_def_id(hir_id));

            let variants = std::iter::once(convert_variant(
                tcx,
                variant_did,
                ctor_did,
                item.ident,
                ty::VariantDiscr::Relative(0),
                def,
                AdtKind::Union,
                def_id,
            ))
            .collect();

            (AdtKind::Union, variants)
        }
        _ => bug!(),
    };
    tcx.alloc_adt_def(def_id.to_def_id(), kind, variants, repr)
}

fn trait_def(tcx: TyCtxt<'_>, def_id: DefId) -> ty::TraitDef {
    let item = tcx.hir().expect_item(def_id.expect_local());

    let (is_auto, unsafety, items) = match item.kind {
        hir::ItemKind::Trait(is_auto, unsafety, .., items) => {
            (is_auto == hir::IsAuto::Yes, unsafety, items)
        }
        hir::ItemKind::TraitAlias(..) => (false, hir::Unsafety::Normal, &[][..]),
        _ => span_bug!(item.span, "trait_def_of_item invoked on non-trait"),
    };

    let paren_sugar = tcx.has_attr(def_id, sym::rustc_paren_sugar);
    if paren_sugar && !tcx.features().unboxed_closures {
        tcx.sess
            .struct_span_err(
                item.span,
                "the `#[rustc_paren_sugar]` attribute is a temporary means of controlling \
                 which traits can use parenthetical notation",
            )
            .help("add `#![feature(unboxed_closures)]` to the crate attributes to use it")
            .emit();
    }

    let is_marker = tcx.has_attr(def_id, sym::marker);
    let skip_array_during_method_dispatch =
        tcx.has_attr(def_id, sym::rustc_skip_array_during_method_dispatch);
    let spec_kind = if tcx.has_attr(def_id, sym::rustc_unsafe_specialization_marker) {
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
                tcx.sess
                    .struct_span_err(
                        attr.span,
                        "the `#[rustc_must_implement_one_of]` attribute must be \
                        used with at least 2 args",
                    )
                    .emit();

                None
            }
            Some(items) => items
                .into_iter()
                .map(|item| item.ident().ok_or(item.span()))
                .collect::<Result<Box<[_]>, _>>()
                .map_err(|span| {
                    tcx.sess
                        .struct_span_err(span, "must be a name of an associated function")
                        .emit();
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
                        if !tcx.impl_defaultness(item.id.def_id).has_value() {
                            tcx.sess
                                .struct_span_err(
                                    item.span,
                                    "This function doesn't have a default implementation",
                                )
                                .span_note(attr_span, "required by this annotation")
                                .emit();

                            return Some(());
                        }

                        return None;
                    }
                    Some(item) => {
                        tcx.sess
                            .struct_span_err(item.span, "Not a function")
                            .span_note(attr_span, "required by this annotation")
                            .note(
                                "All `#[rustc_must_implement_one_of]` arguments \
                            must be associated function names",
                            )
                            .emit();
                    }
                    None => {
                        tcx.sess
                            .struct_span_err(ident.span, "Function not found in this trait")
                            .emit();
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
                        .struct_span_err(vec![dup, ident.span], "Functions names are duplicated")
                        .note(
                            "All `#[rustc_must_implement_one_of]` arguments \
                            must be unique",
                        )
                        .emit();

                    no_dups = false;
                }
            }

            no_dups.then_some(list)
        });

    ty::TraitDef::new(
        def_id,
        unsafety,
        paren_sugar,
        is_auto,
        is_marker,
        skip_array_during_method_dispatch,
        spec_kind,
        must_implement_one_of,
    )
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
        Ptr(mut_ty) | Rptr(_, mut_ty) => is_suggestable_infer_ty(mut_ty.ty),
        OpaqueDef(_, generic_args, _) => are_suggestable_generic_args(generic_args),
        Path(hir::QPath::TypeRelative(ty, segment)) => {
            is_suggestable_infer_ty(ty) || are_suggestable_generic_args(segment.args().args)
        }
        Path(hir::QPath::Resolved(ty_opt, hir::Path { segments, .. })) => {
            ty_opt.map_or(false, is_suggestable_infer_ty)
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
fn fn_sig(tcx: TyCtxt<'_>, def_id: DefId) -> ty::PolyFnSig<'_> {
    use rustc_hir::Node::*;
    use rustc_hir::*;

    let def_id = def_id.expect_local();
    let hir_id = tcx.hir().local_def_id_to_hir_id(def_id);

    let icx = ItemCtxt::new(tcx, def_id.to_def_id());

    match tcx.hir().get(hir_id) {
        TraitItem(hir::TraitItem {
            kind: TraitItemKind::Fn(sig, TraitFn::Provided(_)),
            generics,
            ..
        })
        | Item(hir::Item { kind: ItemKind::Fn(sig, generics, _), .. }) => {
            infer_return_ty_for_fn_sig(tcx, sig, generics, def_id, &icx)
        }

        ImplItem(hir::ImplItem { kind: ImplItemKind::Fn(sig, _), generics, .. }) => {
            // Do not try to inference the return type for a impl method coming from a trait
            if let Item(hir::Item { kind: ItemKind::Impl(i), .. }) =
                tcx.hir().get(tcx.hir().get_parent_node(hir_id))
                && i.of_trait.is_some()
            {
                <dyn AstConv<'_>>::ty_of_fn(
                    &icx,
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
        }) => <dyn AstConv<'_>>::ty_of_fn(
            &icx,
            hir_id,
            header.unsafety,
            header.abi,
            decl,
            Some(generics),
            None,
        ),

        ForeignItem(&hir::ForeignItem { kind: ForeignItemKind::Fn(fn_decl, _, _), .. }) => {
            let abi = tcx.hir().get_foreign_abi(hir_id);
            compute_sig_of_foreign_fn_decl(tcx, def_id.to_def_id(), fn_decl, abi)
        }

        Ctor(data) | Variant(hir::Variant { data, .. }) if data.ctor_hir_id().is_some() => {
            let ty = tcx.type_of(tcx.hir().get_parent_item(hir_id));
            let inputs =
                data.fields().iter().map(|f| tcx.type_of(tcx.hir().local_def_id(f.hir_id)));
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
    }
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
            let fn_sig = ty::Binder::dummy(fn_sig);

            let mut visitor = HirPlaceholderCollector::default();
            visitor.visit_ty(ty);
            let mut diag = bad_placeholder(tcx, visitor.0, "return type");
            let ret_ty = fn_sig.skip_binder().output();
            if ret_ty.is_suggestable(tcx, false) {
                diag.span_suggestion(
                    ty.span,
                    "replace with the correct return type",
                    ret_ty,
                    Applicability::MachineApplicable,
                );
            } else if matches!(ret_ty.kind(), ty::FnDef(..)) {
                let fn_sig = ret_ty.fn_sig(tcx);
                if fn_sig
                    .skip_binder()
                    .inputs_and_output
                    .iter()
                    .all(|t| t.is_suggestable(tcx, false))
                {
                    diag.span_suggestion(
                        ty.span,
                        "replace with the correct return type",
                        fn_sig,
                        Applicability::MachineApplicable,
                    );
                }
            } else if ret_ty.is_closure() {
                // We're dealing with a closure, so we should suggest using `impl Fn` or trait bounds
                // to prevent the user from getting a papercut while trying to use the unique closure
                // syntax (e.g. `[closure@src/lib.rs:2:5: 2:9]`).
                diag.help("consider using an `Fn`, `FnMut`, or `FnOnce` trait bound");
                diag.note("for more information on `Fn` traits and closure types, see https://doc.rust-lang.org/book/ch13-01-closures.html");
            }
            diag.emit();

            fn_sig
        }
        None => <dyn AstConv<'_>>::ty_of_fn(
            icx,
            hir_id,
            sig.header.unsafety,
            sig.header.abi,
            sig.decl,
            Some(generics),
            None,
        ),
    }
}

fn impl_trait_ref(tcx: TyCtxt<'_>, def_id: DefId) -> Option<ty::TraitRef<'_>> {
    let icx = ItemCtxt::new(tcx, def_id);
    match tcx.hir().expect_item(def_id.expect_local()).kind {
        hir::ItemKind::Impl(ref impl_) => impl_.of_trait.as_ref().map(|ast_trait_ref| {
            let selfty = tcx.type_of(def_id);
            <dyn AstConv<'_>>::instantiate_mono_trait_ref(&icx, ast_trait_ref, selfty)
        }),
        _ => bug!(),
    }
}

fn impl_polarity(tcx: TyCtxt<'_>, def_id: DefId) -> ty::ImplPolarity {
    let is_rustc_reservation = tcx.has_attr(def_id, sym::rustc_reservation_impl);
    let item = tcx.hir().expect_item(def_id.expect_local());
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
        if result.predicates.is_empty() {
            result.predicates = inferred_outlives;
        } else {
            result.predicates = tcx
                .arena
                .alloc_from_iter(result.predicates.iter().chain(inferred_outlives).copied());
        }
    }

    debug!("predicates_defined_on({:?}) = {:?}", def_id, result);
    result
}

fn compute_sig_of_foreign_fn_decl<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: DefId,
    decl: &'tcx hir::FnDecl<'tcx>,
    abi: abi::Abi,
) -> ty::PolyFnSig<'tcx> {
    let unsafety = if abi == abi::Abi::RustIntrinsic {
        intrinsic_operation_unsafety(tcx, def_id)
    } else {
        hir::Unsafety::Unsafe
    };
    let hir_id = tcx.hir().local_def_id_to_hir_id(def_id.expect_local());
    let fty = <dyn AstConv<'_>>::ty_of_fn(
        &ItemCtxt::new(tcx, def_id),
        hir_id,
        unsafety,
        abi,
        decl,
        None,
        None,
    );

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
                tcx.sess
                    .struct_span_err(
                        ast_ty.span,
                        &format!(
                            "use of SIMD type{} in FFI is highly experimental and \
                             may result in invalid code",
                            snip
                        ),
                    )
                    .help("add `#![feature(simd_ffi)]` to the crate attributes to enable")
                    .emit();
            }
        };
        for (input, ty) in iter::zip(decl.inputs, fty.inputs().skip_binder()) {
            check(input, *ty)
        }
        if let hir::FnRetTy::Return(ref ty) = decl.output {
            check(ty, fty.output().skip_binder())
        }
    }

    fty
}

fn is_foreign_item(tcx: TyCtxt<'_>, def_id: DefId) -> bool {
    match tcx.hir().get_if_local(def_id) {
        Some(Node::ForeignItem(..)) => true,
        Some(_) => false,
        _ => bug!("is_foreign_item applied to non-local def-id {:?}", def_id),
    }
}

fn generator_kind(tcx: TyCtxt<'_>, def_id: DefId) -> Option<hir::GeneratorKind> {
    match tcx.hir().get_if_local(def_id) {
        Some(Node::Expr(&rustc_hir::Expr {
            kind: rustc_hir::ExprKind::Closure(&rustc_hir::Closure { body, .. }),
            ..
        })) => tcx.hir().body(body).generator_kind(),
        Some(_) => None,
        _ => bug!("generator_kind applied to non-local def-id {:?}", def_id),
    }
}

fn from_target_feature(
    tcx: TyCtxt<'_>,
    attr: &ast::Attribute,
    supported_target_features: &FxHashMap<String, Option<Symbol>>,
    target_features: &mut Vec<Symbol>,
) {
    let Some(list) = attr.meta_item_list() else { return };
    let bad_item = |span| {
        let msg = "malformed `target_feature` attribute input";
        let code = "enable = \"..\"";
        tcx.sess
            .struct_span_err(span, msg)
            .span_suggestion(span, "must be of the form", code, Applicability::HasPlaceholders)
            .emit();
    };
    let rust_features = tcx.features();
    for item in list {
        // Only `enable = ...` is accepted in the meta-item list.
        if !item.has_name(sym::enable) {
            bad_item(item.span());
            continue;
        }

        // Must be of the form `enable = "..."` (a string).
        let Some(value) = item.value_str() else {
            bad_item(item.span());
            continue;
        };

        // We allow comma separation to enable multiple features.
        target_features.extend(value.as_str().split(',').filter_map(|feature| {
            let Some(feature_gate) = supported_target_features.get(feature) else {
                let msg =
                    format!("the feature named `{}` is not valid for this target", feature);
                let mut err = tcx.sess.struct_span_err(item.span(), &msg);
                err.span_label(
                    item.span(),
                    format!("`{}` is not valid for this target", feature),
                );
                if let Some(stripped) = feature.strip_prefix('+') {
                    let valid = supported_target_features.contains_key(stripped);
                    if valid {
                        err.help("consider removing the leading `+` in the feature name");
                    }
                }
                err.emit();
                return None;
            };

            // Only allow features whose feature gates have been enabled.
            let allowed = match feature_gate.as_ref().copied() {
                Some(sym::arm_target_feature) => rust_features.arm_target_feature,
                Some(sym::hexagon_target_feature) => rust_features.hexagon_target_feature,
                Some(sym::powerpc_target_feature) => rust_features.powerpc_target_feature,
                Some(sym::mips_target_feature) => rust_features.mips_target_feature,
                Some(sym::riscv_target_feature) => rust_features.riscv_target_feature,
                Some(sym::avx512_target_feature) => rust_features.avx512_target_feature,
                Some(sym::sse4a_target_feature) => rust_features.sse4a_target_feature,
                Some(sym::tbm_target_feature) => rust_features.tbm_target_feature,
                Some(sym::wasm_target_feature) => rust_features.wasm_target_feature,
                Some(sym::cmpxchg16b_target_feature) => rust_features.cmpxchg16b_target_feature,
                Some(sym::movbe_target_feature) => rust_features.movbe_target_feature,
                Some(sym::rtm_target_feature) => rust_features.rtm_target_feature,
                Some(sym::f16c_target_feature) => rust_features.f16c_target_feature,
                Some(sym::ermsb_target_feature) => rust_features.ermsb_target_feature,
                Some(sym::bpf_target_feature) => rust_features.bpf_target_feature,
                Some(sym::aarch64_ver_target_feature) => rust_features.aarch64_ver_target_feature,
                Some(name) => bug!("unknown target feature gate {}", name),
                None => true,
            };
            if !allowed {
                feature_err(
                    &tcx.sess.parse_sess,
                    feature_gate.unwrap(),
                    item.span(),
                    &format!("the target feature `{}` is currently unstable", feature),
                )
                .emit();
            }
            Some(Symbol::intern(feature))
        }));
    }
}

fn linkage_by_name(tcx: TyCtxt<'_>, def_id: LocalDefId, name: &str) -> Linkage {
    use rustc_middle::mir::mono::Linkage::*;

    // Use the names from src/llvm/docs/LangRef.rst here. Most types are only
    // applicable to variable declarations and may not really make sense for
    // Rust code in the first place but allow them anyway and trust that the
    // user knows what they're doing. Who knows, unanticipated use cases may pop
    // up in the future.
    //
    // ghost, dllimport, dllexport and linkonce_odr_autohide are not supported
    // and don't have to be, LLVM treats them as no-ops.
    match name {
        "appending" => Appending,
        "available_externally" => AvailableExternally,
        "common" => Common,
        "extern_weak" => ExternalWeak,
        "external" => External,
        "internal" => Internal,
        "linkonce" => LinkOnceAny,
        "linkonce_odr" => LinkOnceODR,
        "private" => Private,
        "weak" => WeakAny,
        "weak_odr" => WeakODR,
        _ => tcx.sess.span_fatal(tcx.def_span(def_id), "invalid linkage specified"),
    }
}

fn codegen_fn_attrs(tcx: TyCtxt<'_>, did: DefId) -> CodegenFnAttrs {
    if cfg!(debug_assertions) {
        let def_kind = tcx.def_kind(did);
        assert!(
            def_kind.has_codegen_attrs(),
            "unexpected `def_kind` in `codegen_fn_attrs`: {def_kind:?}",
        );
    }

    let did = did.expect_local();
    let attrs = tcx.hir().attrs(tcx.hir().local_def_id_to_hir_id(did));
    let mut codegen_fn_attrs = CodegenFnAttrs::new();
    if tcx.should_inherit_track_caller(did) {
        codegen_fn_attrs.flags |= CodegenFnAttrFlags::TRACK_CALLER;
    }

    // The panic_no_unwind function called by TerminatorKind::Abort will never
    // unwind. If the panic handler that it invokes unwind then it will simply
    // call the panic handler again.
    if Some(did.to_def_id()) == tcx.lang_items().panic_no_unwind() {
        codegen_fn_attrs.flags |= CodegenFnAttrFlags::NEVER_UNWIND;
    }

    let supported_target_features = tcx.supported_target_features(LOCAL_CRATE);

    let mut inline_span = None;
    let mut link_ordinal_span = None;
    let mut no_sanitize_span = None;
    for attr in attrs.iter() {
        if attr.has_name(sym::cold) {
            codegen_fn_attrs.flags |= CodegenFnAttrFlags::COLD;
        } else if attr.has_name(sym::rustc_allocator) {
            codegen_fn_attrs.flags |= CodegenFnAttrFlags::ALLOCATOR;
        } else if attr.has_name(sym::ffi_returns_twice) {
            if tcx.is_foreign_item(did) {
                codegen_fn_attrs.flags |= CodegenFnAttrFlags::FFI_RETURNS_TWICE;
            } else {
                // `#[ffi_returns_twice]` is only allowed `extern fn`s.
                struct_span_err!(
                    tcx.sess,
                    attr.span,
                    E0724,
                    "`#[ffi_returns_twice]` may only be used on foreign functions"
                )
                .emit();
            }
        } else if attr.has_name(sym::ffi_pure) {
            if tcx.is_foreign_item(did) {
                if attrs.iter().any(|a| a.has_name(sym::ffi_const)) {
                    // `#[ffi_const]` functions cannot be `#[ffi_pure]`
                    struct_span_err!(
                        tcx.sess,
                        attr.span,
                        E0757,
                        "`#[ffi_const]` function cannot be `#[ffi_pure]`"
                    )
                    .emit();
                } else {
                    codegen_fn_attrs.flags |= CodegenFnAttrFlags::FFI_PURE;
                }
            } else {
                // `#[ffi_pure]` is only allowed on foreign functions
                struct_span_err!(
                    tcx.sess,
                    attr.span,
                    E0755,
                    "`#[ffi_pure]` may only be used on foreign functions"
                )
                .emit();
            }
        } else if attr.has_name(sym::ffi_const) {
            if tcx.is_foreign_item(did) {
                codegen_fn_attrs.flags |= CodegenFnAttrFlags::FFI_CONST;
            } else {
                // `#[ffi_const]` is only allowed on foreign functions
                struct_span_err!(
                    tcx.sess,
                    attr.span,
                    E0756,
                    "`#[ffi_const]` may only be used on foreign functions"
                )
                .emit();
            }
        } else if attr.has_name(sym::rustc_allocator_nounwind) {
            codegen_fn_attrs.flags |= CodegenFnAttrFlags::NEVER_UNWIND;
        } else if attr.has_name(sym::rustc_reallocator) {
            codegen_fn_attrs.flags |= CodegenFnAttrFlags::REALLOCATOR;
        } else if attr.has_name(sym::rustc_deallocator) {
            codegen_fn_attrs.flags |= CodegenFnAttrFlags::DEALLOCATOR;
        } else if attr.has_name(sym::rustc_allocator_zeroed) {
            codegen_fn_attrs.flags |= CodegenFnAttrFlags::ALLOCATOR_ZEROED;
        } else if attr.has_name(sym::naked) {
            codegen_fn_attrs.flags |= CodegenFnAttrFlags::NAKED;
        } else if attr.has_name(sym::no_mangle) {
            codegen_fn_attrs.flags |= CodegenFnAttrFlags::NO_MANGLE;
        } else if attr.has_name(sym::no_coverage) {
            codegen_fn_attrs.flags |= CodegenFnAttrFlags::NO_COVERAGE;
        } else if attr.has_name(sym::rustc_std_internal_symbol) {
            codegen_fn_attrs.flags |= CodegenFnAttrFlags::RUSTC_STD_INTERNAL_SYMBOL;
        } else if attr.has_name(sym::used) {
            let inner = attr.meta_item_list();
            match inner.as_deref() {
                Some([item]) if item.has_name(sym::linker) => {
                    if !tcx.features().used_with_arg {
                        feature_err(
                            &tcx.sess.parse_sess,
                            sym::used_with_arg,
                            attr.span,
                            "`#[used(linker)]` is currently unstable",
                        )
                        .emit();
                    }
                    codegen_fn_attrs.flags |= CodegenFnAttrFlags::USED_LINKER;
                }
                Some([item]) if item.has_name(sym::compiler) => {
                    if !tcx.features().used_with_arg {
                        feature_err(
                            &tcx.sess.parse_sess,
                            sym::used_with_arg,
                            attr.span,
                            "`#[used(compiler)]` is currently unstable",
                        )
                        .emit();
                    }
                    codegen_fn_attrs.flags |= CodegenFnAttrFlags::USED;
                }
                Some(_) => {
                    tcx.sess.emit_err(errors::ExpectedUsedSymbol { span: attr.span });
                }
                None => {
                    // Unfortunately, unconditionally using `llvm.used` causes
                    // issues in handling `.init_array` with the gold linker,
                    // but using `llvm.compiler.used` caused a nontrival amount
                    // of unintentional ecosystem breakage -- particularly on
                    // Mach-O targets.
                    //
                    // As a result, we emit `llvm.compiler.used` only on ELF
                    // targets. This is somewhat ad-hoc, but actually follows
                    // our pre-LLVM 13 behavior (prior to the ecosystem
                    // breakage), and seems to match `clang`'s behavior as well
                    // (both before and after LLVM 13), possibly because they
                    // have similar compatibility concerns to us. See
                    // https://github.com/rust-lang/rust/issues/47384#issuecomment-1019080146
                    // and following comments for some discussion of this, as
                    // well as the comments in `rustc_codegen_llvm` where these
                    // flags are handled.
                    //
                    // Anyway, to be clear: this is still up in the air
                    // somewhat, and is subject to change in the future (which
                    // is a good thing, because this would ideally be a bit
                    // more firmed up).
                    let is_like_elf = !(tcx.sess.target.is_like_osx
                        || tcx.sess.target.is_like_windows
                        || tcx.sess.target.is_like_wasm);
                    codegen_fn_attrs.flags |= if is_like_elf {
                        CodegenFnAttrFlags::USED
                    } else {
                        CodegenFnAttrFlags::USED_LINKER
                    };
                }
            }
        } else if attr.has_name(sym::cmse_nonsecure_entry) {
            if !matches!(tcx.fn_sig(did).abi(), abi::Abi::C { .. }) {
                struct_span_err!(
                    tcx.sess,
                    attr.span,
                    E0776,
                    "`#[cmse_nonsecure_entry]` requires C ABI"
                )
                .emit();
            }
            if !tcx.sess.target.llvm_target.contains("thumbv8m") {
                struct_span_err!(tcx.sess, attr.span, E0775, "`#[cmse_nonsecure_entry]` is only valid for targets with the TrustZone-M extension")
                    .emit();
            }
            codegen_fn_attrs.flags |= CodegenFnAttrFlags::CMSE_NONSECURE_ENTRY;
        } else if attr.has_name(sym::thread_local) {
            codegen_fn_attrs.flags |= CodegenFnAttrFlags::THREAD_LOCAL;
        } else if attr.has_name(sym::track_caller) {
            if !tcx.is_closure(did.to_def_id()) && tcx.fn_sig(did).abi() != abi::Abi::Rust {
                struct_span_err!(tcx.sess, attr.span, E0737, "`#[track_caller]` requires Rust ABI")
                    .emit();
            }
            if tcx.is_closure(did.to_def_id()) && !tcx.features().closure_track_caller {
                feature_err(
                    &tcx.sess.parse_sess,
                    sym::closure_track_caller,
                    attr.span,
                    "`#[track_caller]` on closures is currently unstable",
                )
                .emit();
            }
            codegen_fn_attrs.flags |= CodegenFnAttrFlags::TRACK_CALLER;
        } else if attr.has_name(sym::export_name) {
            if let Some(s) = attr.value_str() {
                if s.as_str().contains('\0') {
                    // `#[export_name = ...]` will be converted to a null-terminated string,
                    // so it may not contain any null characters.
                    struct_span_err!(
                        tcx.sess,
                        attr.span,
                        E0648,
                        "`export_name` may not contain null characters"
                    )
                    .emit();
                }
                codegen_fn_attrs.export_name = Some(s);
            }
        } else if attr.has_name(sym::target_feature) {
            if !tcx.is_closure(did.to_def_id())
                && tcx.fn_sig(did).unsafety() == hir::Unsafety::Normal
            {
                if tcx.sess.target.is_like_wasm || tcx.sess.opts.actually_rustdoc {
                    // The `#[target_feature]` attribute is allowed on
                    // WebAssembly targets on all functions, including safe
                    // ones. Other targets require that `#[target_feature]` is
                    // only applied to unsafe functions (pending the
                    // `target_feature_11` feature) because on most targets
                    // execution of instructions that are not supported is
                    // considered undefined behavior. For WebAssembly which is a
                    // 100% safe target at execution time it's not possible to
                    // execute undefined instructions, and even if a future
                    // feature was added in some form for this it would be a
                    // deterministic trap. There is no undefined behavior when
                    // executing WebAssembly so `#[target_feature]` is allowed
                    // on safe functions (but again, only for WebAssembly)
                    //
                    // Note that this is also allowed if `actually_rustdoc` so
                    // if a target is documenting some wasm-specific code then
                    // it's not spuriously denied.
                } else if !tcx.features().target_feature_11 {
                    let mut err = feature_err(
                        &tcx.sess.parse_sess,
                        sym::target_feature_11,
                        attr.span,
                        "`#[target_feature(..)]` can only be applied to `unsafe` functions",
                    );
                    err.span_label(tcx.def_span(did), "not an `unsafe` function");
                    err.emit();
                } else {
                    check_target_feature_trait_unsafe(tcx, did, attr.span);
                }
            }
            from_target_feature(
                tcx,
                attr,
                supported_target_features,
                &mut codegen_fn_attrs.target_features,
            );
        } else if attr.has_name(sym::linkage) {
            if let Some(val) = attr.value_str() {
                codegen_fn_attrs.linkage = Some(linkage_by_name(tcx, did, val.as_str()));
            }
        } else if attr.has_name(sym::link_section) {
            if let Some(val) = attr.value_str() {
                if val.as_str().bytes().any(|b| b == 0) {
                    let msg = format!(
                        "illegal null byte in link_section \
                         value: `{}`",
                        &val
                    );
                    tcx.sess.span_err(attr.span, &msg);
                } else {
                    codegen_fn_attrs.link_section = Some(val);
                }
            }
        } else if attr.has_name(sym::link_name) {
            codegen_fn_attrs.link_name = attr.value_str();
        } else if attr.has_name(sym::link_ordinal) {
            link_ordinal_span = Some(attr.span);
            if let ordinal @ Some(_) = check_link_ordinal(tcx, attr) {
                codegen_fn_attrs.link_ordinal = ordinal;
            }
        } else if attr.has_name(sym::no_sanitize) {
            no_sanitize_span = Some(attr.span);
            if let Some(list) = attr.meta_item_list() {
                for item in list.iter() {
                    if item.has_name(sym::address) {
                        codegen_fn_attrs.no_sanitize |= SanitizerSet::ADDRESS;
                    } else if item.has_name(sym::cfi) {
                        codegen_fn_attrs.no_sanitize |= SanitizerSet::CFI;
                    } else if item.has_name(sym::memory) {
                        codegen_fn_attrs.no_sanitize |= SanitizerSet::MEMORY;
                    } else if item.has_name(sym::memtag) {
                        codegen_fn_attrs.no_sanitize |= SanitizerSet::MEMTAG;
                    } else if item.has_name(sym::shadow_call_stack) {
                        codegen_fn_attrs.no_sanitize |= SanitizerSet::SHADOWCALLSTACK;
                    } else if item.has_name(sym::thread) {
                        codegen_fn_attrs.no_sanitize |= SanitizerSet::THREAD;
                    } else if item.has_name(sym::hwaddress) {
                        codegen_fn_attrs.no_sanitize |= SanitizerSet::HWADDRESS;
                    } else {
                        tcx.sess
                            .struct_span_err(item.span(), "invalid argument for `no_sanitize`")
                            .note("expected one of: `address`, `cfi`, `hwaddress`, `memory`, `memtag`, `shadow-call-stack`, or `thread`")
                            .emit();
                    }
                }
            }
        } else if attr.has_name(sym::instruction_set) {
            codegen_fn_attrs.instruction_set = match attr.meta_kind() {
                Some(MetaItemKind::List(ref items)) => match items.as_slice() {
                    [NestedMetaItem::MetaItem(set)] => {
                        let segments =
                            set.path.segments.iter().map(|x| x.ident.name).collect::<Vec<_>>();
                        match segments.as_slice() {
                            [sym::arm, sym::a32] | [sym::arm, sym::t32] => {
                                if !tcx.sess.target.has_thumb_interworking {
                                    struct_span_err!(
                                        tcx.sess.diagnostic(),
                                        attr.span,
                                        E0779,
                                        "target does not support `#[instruction_set]`"
                                    )
                                    .emit();
                                    None
                                } else if segments[1] == sym::a32 {
                                    Some(InstructionSetAttr::ArmA32)
                                } else if segments[1] == sym::t32 {
                                    Some(InstructionSetAttr::ArmT32)
                                } else {
                                    unreachable!()
                                }
                            }
                            _ => {
                                struct_span_err!(
                                    tcx.sess.diagnostic(),
                                    attr.span,
                                    E0779,
                                    "invalid instruction set specified",
                                )
                                .emit();
                                None
                            }
                        }
                    }
                    [] => {
                        struct_span_err!(
                            tcx.sess.diagnostic(),
                            attr.span,
                            E0778,
                            "`#[instruction_set]` requires an argument"
                        )
                        .emit();
                        None
                    }
                    _ => {
                        struct_span_err!(
                            tcx.sess.diagnostic(),
                            attr.span,
                            E0779,
                            "cannot specify more than one instruction set"
                        )
                        .emit();
                        None
                    }
                },
                _ => {
                    struct_span_err!(
                        tcx.sess.diagnostic(),
                        attr.span,
                        E0778,
                        "must specify an instruction set"
                    )
                    .emit();
                    None
                }
            };
        } else if attr.has_name(sym::repr) {
            codegen_fn_attrs.alignment = match attr.meta_item_list() {
                Some(items) => match items.as_slice() {
                    [item] => match item.name_value_literal() {
                        Some((sym::align, literal)) => {
                            let alignment = rustc_attr::parse_alignment(&literal.kind);

                            match alignment {
                                Ok(align) => Some(align),
                                Err(msg) => {
                                    struct_span_err!(
                                        tcx.sess.diagnostic(),
                                        attr.span,
                                        E0589,
                                        "invalid `repr(align)` attribute: {}",
                                        msg
                                    )
                                    .emit();

                                    None
                                }
                            }
                        }
                        _ => None,
                    },
                    [] => None,
                    _ => None,
                },
                None => None,
            };
        }
    }

    codegen_fn_attrs.inline = attrs.iter().fold(InlineAttr::None, |ia, attr| {
        if !attr.has_name(sym::inline) {
            return ia;
        }
        match attr.meta_kind() {
            Some(MetaItemKind::Word) => InlineAttr::Hint,
            Some(MetaItemKind::List(ref items)) => {
                inline_span = Some(attr.span);
                if items.len() != 1 {
                    struct_span_err!(
                        tcx.sess.diagnostic(),
                        attr.span,
                        E0534,
                        "expected one argument"
                    )
                    .emit();
                    InlineAttr::None
                } else if list_contains_name(&items, sym::always) {
                    InlineAttr::Always
                } else if list_contains_name(&items, sym::never) {
                    InlineAttr::Never
                } else {
                    struct_span_err!(
                        tcx.sess.diagnostic(),
                        items[0].span(),
                        E0535,
                        "invalid argument"
                    )
                    .help("valid inline arguments are `always` and `never`")
                    .emit();

                    InlineAttr::None
                }
            }
            Some(MetaItemKind::NameValue(_)) => ia,
            None => ia,
        }
    });

    codegen_fn_attrs.optimize = attrs.iter().fold(OptimizeAttr::None, |ia, attr| {
        if !attr.has_name(sym::optimize) {
            return ia;
        }
        let err = |sp, s| struct_span_err!(tcx.sess.diagnostic(), sp, E0722, "{}", s).emit();
        match attr.meta_kind() {
            Some(MetaItemKind::Word) => {
                err(attr.span, "expected one argument");
                ia
            }
            Some(MetaItemKind::List(ref items)) => {
                inline_span = Some(attr.span);
                if items.len() != 1 {
                    err(attr.span, "expected one argument");
                    OptimizeAttr::None
                } else if list_contains_name(&items, sym::size) {
                    OptimizeAttr::Size
                } else if list_contains_name(&items, sym::speed) {
                    OptimizeAttr::Speed
                } else {
                    err(items[0].span(), "invalid argument");
                    OptimizeAttr::None
                }
            }
            Some(MetaItemKind::NameValue(_)) => ia,
            None => ia,
        }
    });

    // #73631: closures inherit `#[target_feature]` annotations
    if tcx.features().target_feature_11 && tcx.is_closure(did.to_def_id()) {
        let owner_id = tcx.parent(did.to_def_id());
        if tcx.def_kind(owner_id).has_codegen_attrs() {
            codegen_fn_attrs
                .target_features
                .extend(tcx.codegen_fn_attrs(owner_id).target_features.iter().copied());
        }
    }

    // If a function uses #[target_feature] it can't be inlined into general
    // purpose functions as they wouldn't have the right target features
    // enabled. For that reason we also forbid #[inline(always)] as it can't be
    // respected.
    if !codegen_fn_attrs.target_features.is_empty() {
        if codegen_fn_attrs.inline == InlineAttr::Always {
            if let Some(span) = inline_span {
                tcx.sess.span_err(
                    span,
                    "cannot use `#[inline(always)]` with \
                     `#[target_feature]`",
                );
            }
        }
    }

    if !codegen_fn_attrs.no_sanitize.is_empty() {
        if codegen_fn_attrs.inline == InlineAttr::Always {
            if let (Some(no_sanitize_span), Some(inline_span)) = (no_sanitize_span, inline_span) {
                let hir_id = tcx.hir().local_def_id_to_hir_id(did);
                tcx.struct_span_lint_hir(
                    lint::builtin::INLINE_NO_SANITIZE,
                    hir_id,
                    no_sanitize_span,
                    "`no_sanitize` will have no effect after inlining",
                    |lint| lint.span_note(inline_span, "inlining requested here"),
                )
            }
        }
    }

    // Weak lang items have the same semantics as "std internal" symbols in the
    // sense that they're preserved through all our LTO passes and only
    // strippable by the linker.
    //
    // Additionally weak lang items have predetermined symbol names.
    if tcx.is_weak_lang_item(did.to_def_id()) {
        codegen_fn_attrs.flags |= CodegenFnAttrFlags::RUSTC_STD_INTERNAL_SYMBOL;
    }
    if let Some(name) = weak_lang_items::link_name(attrs) {
        codegen_fn_attrs.export_name = Some(name);
        codegen_fn_attrs.link_name = Some(name);
    }
    check_link_name_xor_ordinal(tcx, &codegen_fn_attrs, link_ordinal_span);

    // Internal symbols to the standard library all have no_mangle semantics in
    // that they have defined symbol names present in the function name. This
    // also applies to weak symbols where they all have known symbol names.
    if codegen_fn_attrs.flags.contains(CodegenFnAttrFlags::RUSTC_STD_INTERNAL_SYMBOL) {
        codegen_fn_attrs.flags |= CodegenFnAttrFlags::NO_MANGLE;
    }

    // Any linkage to LLVM intrinsics for now forcibly marks them all as never
    // unwinds since LLVM sometimes can't handle codegen which `invoke`s
    // intrinsic functions.
    if let Some(name) = &codegen_fn_attrs.link_name {
        if name.as_str().starts_with("llvm.") {
            codegen_fn_attrs.flags |= CodegenFnAttrFlags::NEVER_UNWIND;
        }
    }

    codegen_fn_attrs
}

/// Computes the set of target features used in a function for the purposes of
/// inline assembly.
fn asm_target_features<'tcx>(tcx: TyCtxt<'tcx>, did: DefId) -> &'tcx FxHashSet<Symbol> {
    let mut target_features = tcx.sess.unstable_target_features.clone();
    if tcx.def_kind(did).has_codegen_attrs() {
        let attrs = tcx.codegen_fn_attrs(did);
        target_features.extend(&attrs.target_features);
        match attrs.instruction_set {
            None => {}
            Some(InstructionSetAttr::ArmA32) => {
                target_features.remove(&sym::thumb_mode);
            }
            Some(InstructionSetAttr::ArmT32) => {
                target_features.insert(sym::thumb_mode);
            }
        }
    }

    tcx.arena.alloc(target_features)
}

/// Checks if the provided DefId is a method in a trait impl for a trait which has track_caller
/// applied to the method prototype.
fn should_inherit_track_caller(tcx: TyCtxt<'_>, def_id: DefId) -> bool {
    if let Some(impl_item) = tcx.opt_associated_item(def_id)
        && let ty::AssocItemContainer::ImplContainer = impl_item.container
        && let Some(trait_item) = impl_item.trait_item_def_id
    {
        return tcx
            .codegen_fn_attrs(trait_item)
            .flags
            .intersects(CodegenFnAttrFlags::TRACK_CALLER);
    }

    false
}

fn check_link_ordinal(tcx: TyCtxt<'_>, attr: &ast::Attribute) -> Option<u16> {
    use rustc_ast::{Lit, LitIntType, LitKind};
    if !tcx.features().raw_dylib && tcx.sess.target.arch == "x86" {
        feature_err(
            &tcx.sess.parse_sess,
            sym::raw_dylib,
            attr.span,
            "`#[link_ordinal]` is unstable on x86",
        )
        .emit();
    }
    let meta_item_list = attr.meta_item_list();
    let meta_item_list: Option<&[ast::NestedMetaItem]> = meta_item_list.as_ref().map(Vec::as_ref);
    let sole_meta_list = match meta_item_list {
        Some([item]) => item.literal(),
        Some(_) => {
            tcx.sess
                .struct_span_err(attr.span, "incorrect number of arguments to `#[link_ordinal]`")
                .note("the attribute requires exactly one argument")
                .emit();
            return None;
        }
        _ => None,
    };
    if let Some(Lit { kind: LitKind::Int(ordinal, LitIntType::Unsuffixed), .. }) = sole_meta_list {
        // According to the table at https://docs.microsoft.com/en-us/windows/win32/debug/pe-format#import-header,
        // the ordinal must fit into 16 bits.  Similarly, the Ordinal field in COFFShortExport (defined
        // in llvm/include/llvm/Object/COFFImportFile.h), which we use to communicate import information
        // to LLVM for `#[link(kind = "raw-dylib"_])`, is also defined to be uint16_t.
        //
        // FIXME: should we allow an ordinal of 0?  The MSVC toolchain has inconsistent support for this:
        // both LINK.EXE and LIB.EXE signal errors and abort when given a .DEF file that specifies
        // a zero ordinal.  However, llvm-dlltool is perfectly happy to generate an import library
        // for such a .DEF file, and MSVC's LINK.EXE is also perfectly happy to consume an import
        // library produced by LLVM with an ordinal of 0, and it generates an .EXE.  (I don't know yet
        // if the resulting EXE runs, as I haven't yet built the necessary DLL -- see earlier comment
        // about LINK.EXE failing.)
        if *ordinal <= u16::MAX as u128 {
            Some(*ordinal as u16)
        } else {
            let msg = format!("ordinal value in `link_ordinal` is too large: `{}`", &ordinal);
            tcx.sess
                .struct_span_err(attr.span, &msg)
                .note("the value may not exceed `u16::MAX`")
                .emit();
            None
        }
    } else {
        tcx.sess
            .struct_span_err(attr.span, "illegal ordinal format in `link_ordinal`")
            .note("an unsuffixed integer value, e.g., `1`, is expected")
            .emit();
        None
    }
}

fn check_link_name_xor_ordinal(
    tcx: TyCtxt<'_>,
    codegen_fn_attrs: &CodegenFnAttrs,
    inline_span: Option<Span>,
) {
    if codegen_fn_attrs.link_name.is_none() || codegen_fn_attrs.link_ordinal.is_none() {
        return;
    }
    let msg = "cannot use `#[link_name]` with `#[link_ordinal]`";
    if let Some(span) = inline_span {
        tcx.sess.span_err(span, msg);
    } else {
        tcx.sess.err(msg);
    }
}

/// Checks the function annotated with `#[target_feature]` is not a safe
/// trait method implementation, reporting an error if it is.
fn check_target_feature_trait_unsafe(tcx: TyCtxt<'_>, id: LocalDefId, attr_span: Span) {
    let hir_id = tcx.hir().local_def_id_to_hir_id(id);
    let node = tcx.hir().get(hir_id);
    if let Node::ImplItem(hir::ImplItem { kind: hir::ImplItemKind::Fn(..), .. }) = node {
        let parent_id = tcx.hir().get_parent_item(hir_id);
        let parent_item = tcx.hir().expect_item(parent_id.def_id);
        if let hir::ItemKind::Impl(hir::Impl { of_trait: Some(_), .. }) = parent_item.kind {
            tcx.sess
                .struct_span_err(
                    attr_span,
                    "`#[target_feature(..)]` cannot be applied to safe trait method",
                )
                .span_label(attr_span, "cannot be applied to safe trait method")
                .span_label(tcx.def_span(id), "not an `unsafe` function")
                .emit();
        }
    }
}
