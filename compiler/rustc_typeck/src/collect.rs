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
use crate::bounds::Bounds;
use crate::check::intrinsic::intrinsic_operation_unsafety;
use crate::constrained_generic_params as cgp;
use crate::errors;
use crate::middle::resolve_lifetime as rl;
use rustc_ast as ast;
use rustc_ast::Attribute;
use rustc_ast::{MetaItemKind, NestedMetaItem};
use rustc_attr::{list_contains_name, InlineAttr, InstructionSetAttr, OptimizeAttr};
use rustc_data_structures::captures::Captures;
use rustc_data_structures::fx::{FxHashMap, FxHashSet, FxIndexSet};
use rustc_errors::{struct_span_err, Applicability};
use rustc_hir as hir;
use rustc_hir::def::{CtorKind, DefKind, Res};
use rustc_hir::def_id::{DefId, LocalDefId, LOCAL_CRATE};
use rustc_hir::intravisit::{self, NestedVisitorMap, Visitor};
use rustc_hir::weak_lang_items;
use rustc_hir::{GenericParamKind, HirId, Node};
use rustc_middle::hir::map::Map;
use rustc_middle::middle::codegen_fn_attrs::{CodegenFnAttrFlags, CodegenFnAttrs};
use rustc_middle::mir::mono::Linkage;
use rustc_middle::ty::query::Providers;
use rustc_middle::ty::subst::InternalSubsts;
use rustc_middle::ty::util::Discr;
use rustc_middle::ty::util::IntTypeExt;
use rustc_middle::ty::{self, AdtKind, Const, DefIdTree, Ty, TyCtxt};
use rustc_middle::ty::{ReprOptions, ToPredicate, WithConstness};
use rustc_session::lint;
use rustc_session::parse::feature_err;
use rustc_span::symbol::{kw, sym, Ident, Symbol};
use rustc_span::{Span, DUMMY_SP};
use rustc_target::spec::{abi, PanicStrategy, SanitizerSet};
use rustc_trait_selection::traits::error_reporting::suggestions::NextTypeParamName;
use std::iter;

mod item_bounds;
mod type_of;

struct OnlySelfBounds(bool);

///////////////////////////////////////////////////////////////////////////
// Main entry point

fn collect_mod_item_types(tcx: TyCtxt<'_>, module_def_id: LocalDefId) {
    tcx.hir().visit_item_likes_in_module(
        module_def_id,
        &mut CollectItemTypesVisitor { tcx }.as_deep_visitor(),
    );
}

pub fn provide(providers: &mut Providers) {
    *providers = Providers {
        opt_const_param_of: type_of::opt_const_param_of,
        default_anon_const_substs: type_of::default_anon_const_substs,
        type_of: type_of::type_of,
        item_bounds: item_bounds::item_bounds,
        explicit_item_bounds: item_bounds::explicit_item_bounds,
        generics_of,
        predicates_of,
        predicates_defined_on,
        explicit_predicates_of,
        super_predicates_of,
        super_predicates_that_define_assoc_type,
        trait_explicit_predicates_and_bounds,
        type_param_predicates,
        trait_def,
        adt_def,
        fn_sig,
        impl_trait_ref,
        impl_polarity,
        is_foreign_item,
        static_mutability,
        generator_kind,
        codegen_fn_attrs,
        collect_mod_item_types,
        should_inherit_track_caller,
        ..*providers
    };
}

///////////////////////////////////////////////////////////////////////////

/// Context specific to some particular item. This is what implements
/// `AstConv`. It has information about the predicates that are defined
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
crate struct PlaceholderHirTyCollector(crate Vec<Span>);

impl<'v> Visitor<'v> for PlaceholderHirTyCollector {
    type Map = intravisit::ErasedMap<'v>;

    fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
        NestedVisitorMap::None
    }
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
}

struct CollectItemTypesVisitor<'tcx> {
    tcx: TyCtxt<'tcx>,
}

/// If there are any placeholder types (`_`), emit an error explaining that this is not allowed
/// and suggest adding type parameters in the appropriate place, taking into consideration any and
/// all already existing generic type parameters to avoid suggesting a name that is already in use.
crate fn placeholder_type_error(
    tcx: TyCtxt<'tcx>,
    span: Option<Span>,
    generics: &[hir::GenericParam<'_>],
    placeholder_types: Vec<Span>,
    suggest: bool,
    hir_ty: Option<&hir::Ty<'_>>,
    kind: &'static str,
) {
    if placeholder_types.is_empty() {
        return;
    }

    let type_name = generics.next_type_param_name(None);
    let mut sugg: Vec<_> =
        placeholder_types.iter().map(|sp| (*sp, (*type_name).to_string())).collect();

    if generics.is_empty() {
        if let Some(span) = span {
            sugg.push((span, format!("<{}>", type_name)));
        }
    } else if let Some(arg) = generics
        .iter()
        .find(|arg| matches!(arg.name, hir::ParamName::Plain(Ident { name: kw::Underscore, .. })))
    {
        // Account for `_` already present in cases like `struct S<_>(_);` and suggest
        // `struct S<T>(T);` instead of `struct S<_, T>(T);`.
        sugg.push((arg.span, (*type_name).to_string()));
    } else {
        let last = generics.iter().last().unwrap();
        sugg.push((
            // Account for bounds, we want `fn foo<T: E, K>(_: K)` not `fn foo<T, K: E>(_: K)`.
            last.bounds_span().unwrap_or(last.span).shrink_to_hi(),
            format!(", {}", type_name),
        ));
    }

    let mut err = bad_placeholder_type(tcx, placeholder_types, kind);

    // Suggest, but only if it is not a function in const or static
    if suggest {
        let mut is_fn = false;
        let mut is_const_or_static = false;

        if let Some(hir_ty) = hir_ty {
            if let hir::TyKind::BareFn(_) = hir_ty.kind {
                is_fn = true;

                // Check if parent is const or static
                let parent_id = tcx.hir().get_parent_node(hir_ty.hir_id);
                let parent_node = tcx.hir().get(parent_id);

                is_const_or_static = match parent_node {
                    Node::Item(&hir::Item {
                        kind: hir::ItemKind::Const(..) | hir::ItemKind::Static(..),
                        ..
                    })
                    | Node::TraitItem(&hir::TraitItem {
                        kind: hir::TraitItemKind::Const(..),
                        ..
                    })
                    | Node::ImplItem(&hir::ImplItem {
                        kind: hir::ImplItemKind::Const(..), ..
                    }) => true,
                    _ => false,
                };
            }
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
    err.emit();
}

fn reject_placeholder_type_signatures_in_item(tcx: TyCtxt<'tcx>, item: &'tcx hir::Item<'tcx>) {
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

    let mut visitor = PlaceholderHirTyCollector::default();
    visitor.visit_item(item);

    placeholder_type_error(
        tcx,
        Some(generics.span),
        generics.params,
        visitor.0,
        suggest,
        None,
        item.kind.descr(),
    );
}

impl Visitor<'tcx> for CollectItemTypesVisitor<'tcx> {
    type Map = Map<'tcx>;

    fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
        NestedVisitorMap::OnlyBodies(self.tcx.hir())
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
        if let hir::ExprKind::Closure(..) = expr.kind {
            let def_id = self.tcx.hir().local_def_id(expr.hir_id);
            self.tcx.ensure().generics_of(def_id);
            self.tcx.ensure().type_of(def_id);
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

fn bad_placeholder_type(
    tcx: TyCtxt<'tcx>,
    mut spans: Vec<Span>,
    kind: &'static str,
) -> rustc_errors::DiagnosticBuilder<'tcx> {
    let kind = if kind.ends_with('s') { format!("{}es", kind) } else { format!("{}s", kind) };

    spans.sort();
    let mut err = struct_span_err!(
        tcx.sess,
        spans.clone(),
        E0121,
        "the type placeholder `_` is not allowed within types on item signatures for {}",
        kind
    );
    for span in spans {
        err.span_label(span, "not allowed in type signatures");
    }
    err
}

impl ItemCtxt<'tcx> {
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

impl AstConv<'tcx> for ItemCtxt<'tcx> {
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
        self.tcx().ty_error_with_message(span, "bad_placeholder_type")
    }

    fn ct_infer(
        &self,
        ty: Ty<'tcx>,
        _: Option<&ty::GenericParamDef>,
        span: Span,
    ) -> &'tcx Const<'tcx> {
        bad_placeholder_type(self.tcx(), vec![span], "generic").emit();
        // Typeck doesn't expect erased regions to be returned from `type_of`.
        let ty = self.tcx.fold_regions(ty, &mut false, |r, _| match r {
            ty::ReErased => self.tcx.lifetimes.re_static,
            _ => r,
        });
        self.tcx().const_error(ty)
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
                self.tcx,
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
                    let item =
                        self.tcx.hir().expect_item(self.tcx.hir().get_parent_item(self.hir_id()));
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
                                        self.tcx
                                            .replace_late_bound_regions(poly_trait_ref, |_| {
                                                self.tcx.mk_region(ty::ReEarlyBound(
                                                    ty::EarlyBoundRegion {
                                                        def_id: item_def_id,
                                                        index: 0,
                                                        name: Symbol::intern(&lt_name),
                                                    },
                                                ))
                                            })
                                            .0,
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

/// Returns the predicates defined on `item_def_id` of the form
/// `X: Foo` where `X` is the type parameter `def_id`.
fn type_param_predicates(
    tcx: TyCtxt<'_>,
    (item_def_id, def_id, assoc_name): (DefId, LocalDefId, Ident),
) -> ty::GenericPredicates<'_> {
    use rustc_hir::*;

    // In the AST, bounds can derive from two places. Either
    // written inline like `<T: Foo>` or in a where-clause like
    // `where T: Foo`.

    let param_id = tcx.hir().local_def_id_to_hir_id(def_id);
    let param_owner = tcx.hir().ty_param_owner(param_id);
    let param_owner_def_id = tcx.hir().local_def_id(param_owner);
    let generics = tcx.generics_of(param_owner_def_id);
    let index = generics.param_def_id_to_index[&def_id.to_def_id()];
    let ty = tcx.mk_ty_param(index, tcx.hir().ty_param_name(param_id));

    // Don't look for bounds where the type parameter isn't in scope.
    let parent = if item_def_id == param_owner_def_id.to_def_id() {
        None
    } else {
        tcx.generics_of(item_def_id).parent
    };

    let mut result = parent
        .map(|parent| {
            let icx = ItemCtxt::new(tcx, parent);
            icx.get_type_parameter_bounds(DUMMY_SP, def_id.to_def_id(), assoc_name)
        })
        .unwrap_or_default();
    let mut extend = None;

    let item_hir_id = tcx.hir().local_def_id_to_hir_id(item_def_id.expect_local());
    let ast_generics = match tcx.hir().get(item_hir_id) {
        Node::TraitItem(item) => &item.generics,

        Node::ImplItem(item) => &item.generics,

        Node::Item(item) => {
            match item.kind {
                ItemKind::Fn(.., ref generics, _)
                | ItemKind::Impl(hir::Impl { ref generics, .. })
                | ItemKind::TyAlias(_, ref generics)
                | ItemKind::OpaqueTy(OpaqueTy { ref generics, impl_trait_fn: None, .. })
                | ItemKind::Enum(_, ref generics)
                | ItemKind::Struct(_, ref generics)
                | ItemKind::Union(_, ref generics) => generics,
                ItemKind::Trait(_, _, ref generics, ..) => {
                    // Implied `Self: Trait` and supertrait bounds.
                    if param_id == item_hir_id {
                        let identity_trait_ref = ty::TraitRef::identity(tcx, item_def_id);
                        extend =
                            Some((identity_trait_ref.without_const().to_predicate(tcx), item.span));
                    }
                    generics
                }
                _ => return result,
            }
        }

        Node::ForeignItem(item) => match item.kind {
            ForeignItemKind::Fn(_, _, ref generics) => generics,
            _ => return result,
        },

        _ => return result,
    };

    let icx = ItemCtxt::new(tcx, item_def_id);
    let extra_predicates = extend.into_iter().chain(
        icx.type_parameter_bounds_in_generics(
            ast_generics,
            param_id,
            ty,
            OnlySelfBounds(true),
            Some(assoc_name),
        )
        .into_iter()
        .filter(|(predicate, _)| match predicate.kind().skip_binder() {
            ty::PredicateKind::Trait(data) => data.self_ty().is_param(index),
            _ => false,
        }),
    );
    result.predicates =
        tcx.arena.alloc_from_iter(result.predicates.iter().copied().chain(extra_predicates));
    result
}

impl ItemCtxt<'tcx> {
    /// Finds bounds from `hir::Generics`. This requires scanning through the
    /// AST. We do this to avoid having to convert *all* the bounds, which
    /// would create artificial cycles. Instead, we can only convert the
    /// bounds for a type parameter `X` if `X::Foo` is used.
    fn type_parameter_bounds_in_generics(
        &self,
        ast_generics: &'tcx hir::Generics<'tcx>,
        param_id: hir::HirId,
        ty: Ty<'tcx>,
        only_self_bounds: OnlySelfBounds,
        assoc_name: Option<Ident>,
    ) -> Vec<(ty::Predicate<'tcx>, Span)> {
        let from_ty_params = ast_generics
            .params
            .iter()
            .filter_map(|param| match param.kind {
                GenericParamKind::Type { .. } if param.hir_id == param_id => Some(&param.bounds),
                _ => None,
            })
            .flat_map(|bounds| bounds.iter())
            .filter(|b| match assoc_name {
                Some(assoc_name) => self.bound_defines_assoc_item(b, assoc_name),
                None => true,
            })
            .flat_map(|b| predicates_from_bound(self, ty, b));

        let from_where_clauses = ast_generics
            .where_clause
            .predicates
            .iter()
            .filter_map(|wp| match *wp {
                hir::WherePredicate::BoundPredicate(ref bp) => Some(bp),
                _ => None,
            })
            .flat_map(|bp| {
                let bt = if is_param(self.tcx, &bp.bounded_ty, param_id) {
                    Some(ty)
                } else if !only_self_bounds.0 {
                    Some(self.to_ty(&bp.bounded_ty))
                } else {
                    None
                };
                bp.bounds
                    .iter()
                    .filter(|b| match assoc_name {
                        Some(assoc_name) => self.bound_defines_assoc_item(b, assoc_name),
                        None => true,
                    })
                    .filter_map(move |b| bt.map(|bt| (bt, b)))
            })
            .flat_map(|(bt, b)| predicates_from_bound(self, bt, b));

        from_ty_params.chain(from_where_clauses).collect()
    }

    fn bound_defines_assoc_item(&self, b: &hir::GenericBound<'_>, assoc_name: Ident) -> bool {
        debug!("bound_defines_assoc_item(b={:?}, assoc_name={:?})", b, assoc_name);

        match b {
            hir::GenericBound::Trait(poly_trait_ref, _) => {
                let trait_ref = &poly_trait_ref.trait_ref;
                if let Some(trait_did) = trait_ref.trait_def_id() {
                    self.tcx.trait_may_define_assoc_type(trait_did, assoc_name)
                } else {
                    false
                }
            }
            _ => false,
        }
    }
}

/// Tests whether this is the AST for a reference to the type
/// parameter with ID `param_id`. We use this so as to avoid running
/// `ast_ty_to_ty`, because we want to avoid triggering an all-out
/// conversion of the type to avoid inducing unnecessary cycles.
fn is_param(tcx: TyCtxt<'_>, ast_ty: &hir::Ty<'_>, param_id: hir::HirId) -> bool {
    if let hir::TyKind::Path(hir::QPath::Resolved(None, ref path)) = ast_ty.kind {
        match path.res {
            Res::SelfTy(Some(def_id), None) | Res::Def(DefKind::TyParam, def_id) => {
                def_id == tcx.hir().local_def_id(param_id).to_def_id()
            }
            _ => false,
        }
    } else {
        false
    }
}

fn convert_item(tcx: TyCtxt<'_>, item_id: hir::ItemId) {
    let it = tcx.hir().item(item_id);
    debug!("convert: item {} with id {}", it.ident, it.hir_id());
    let def_id = item_id.def_id;

    match it.kind {
        // These don't define types.
        hir::ItemKind::ExternCrate(_)
        | hir::ItemKind::Use(..)
        | hir::ItemKind::Macro(_)
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
                        let mut visitor = PlaceholderHirTyCollector::default();
                        visitor.visit_foreign_item(item);
                        placeholder_type_error(
                            tcx,
                            None,
                            &[],
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
            convert_enum_variant_types(tcx, def_id.to_def_id(), &enum_definition.variants);
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
        hir::ItemKind::OpaqueTy(hir::OpaqueTy { impl_trait_fn: Some(_), .. }) => {}

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
                    // (#75889): Account for `const C: dyn Fn() -> _ = "";`
                    if let hir::TyKind::TraitObject(..) = ty.kind {
                        let mut visitor = PlaceholderHirTyCollector::default();
                        visitor.visit_item(it);
                        placeholder_type_error(
                            tcx,
                            None,
                            &[],
                            visitor.0,
                            false,
                            None,
                            it.kind.descr(),
                        );
                    }
                }
                _ => (),
            }
        }
    }
}

fn convert_trait_item(tcx: TyCtxt<'_>, trait_item_id: hir::TraitItemId) {
    let trait_item = tcx.hir().trait_item(trait_item_id);
    tcx.ensure().generics_of(trait_item_id.def_id);

    match trait_item.kind {
        hir::TraitItemKind::Fn(..) => {
            tcx.ensure().type_of(trait_item_id.def_id);
            tcx.ensure().fn_sig(trait_item_id.def_id);
        }

        hir::TraitItemKind::Const(.., Some(_)) => {
            tcx.ensure().type_of(trait_item_id.def_id);
        }

        hir::TraitItemKind::Const(..) => {
            tcx.ensure().type_of(trait_item_id.def_id);
            // Account for `const C: _;`.
            let mut visitor = PlaceholderHirTyCollector::default();
            visitor.visit_trait_item(trait_item);
            placeholder_type_error(tcx, None, &[], visitor.0, false, None, "constant");
        }

        hir::TraitItemKind::Type(_, Some(_)) => {
            tcx.ensure().item_bounds(trait_item_id.def_id);
            tcx.ensure().type_of(trait_item_id.def_id);
            // Account for `type T = _;`.
            let mut visitor = PlaceholderHirTyCollector::default();
            visitor.visit_trait_item(trait_item);
            placeholder_type_error(tcx, None, &[], visitor.0, false, None, "associated type");
        }

        hir::TraitItemKind::Type(_, None) => {
            tcx.ensure().item_bounds(trait_item_id.def_id);
            // #74612: Visit and try to find bad placeholders
            // even if there is no concrete type.
            let mut visitor = PlaceholderHirTyCollector::default();
            visitor.visit_trait_item(trait_item);

            placeholder_type_error(tcx, None, &[], visitor.0, false, None, "associated type");
        }
    };

    tcx.ensure().predicates_of(trait_item_id.def_id);
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
            let mut visitor = PlaceholderHirTyCollector::default();
            visitor.visit_impl_item(impl_item);

            placeholder_type_error(tcx, None, &[], visitor.0, false, None, "associated type");
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
    let repr_type = def.repr.discr_type();
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

            ty::FieldDef { did: fid.to_def_id(), ident: f.ident, vis: tcx.visibility(fid) }
        })
        .collect();
    let recovered = match def {
        hir::VariantData::Struct(_, r) => *r,
        _ => false,
    };
    ty::VariantDef::new(
        ident,
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

fn adt_def(tcx: TyCtxt<'_>, def_id: DefId) -> &ty::AdtDef {
    use rustc_hir::*;

    let def_id = def_id.expect_local();
    let hir_id = tcx.hir().local_def_id_to_hir_id(def_id);
    let item = match tcx.hir().get(hir_id) {
        Node::Item(item) => item,
        _ => bug!(),
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

/// Ensures that the super-predicates of the trait with a `DefId`
/// of `trait_def_id` are converted and stored. This also ensures that
/// the transitive super-predicates are converted.
fn super_predicates_of(tcx: TyCtxt<'_>, trait_def_id: DefId) -> ty::GenericPredicates<'_> {
    debug!("super_predicates(trait_def_id={:?})", trait_def_id);
    tcx.super_predicates_that_define_assoc_type((trait_def_id, None))
}

/// Ensures that the super-predicates of the trait with a `DefId`
/// of `trait_def_id` are converted and stored. This also ensures that
/// the transitive super-predicates are converted.
fn super_predicates_that_define_assoc_type(
    tcx: TyCtxt<'_>,
    (trait_def_id, assoc_name): (DefId, Option<Ident>),
) -> ty::GenericPredicates<'_> {
    debug!(
        "super_predicates_that_define_assoc_type(trait_def_id={:?}, assoc_name={:?})",
        trait_def_id, assoc_name
    );
    if trait_def_id.is_local() {
        debug!("super_predicates_that_define_assoc_type: local trait_def_id={:?}", trait_def_id);
        let trait_hir_id = tcx.hir().local_def_id_to_hir_id(trait_def_id.expect_local());

        let item = match tcx.hir().get(trait_hir_id) {
            Node::Item(item) => item,
            _ => bug!("trait_node_id {} is not an item", trait_hir_id),
        };

        let (generics, bounds) = match item.kind {
            hir::ItemKind::Trait(.., ref generics, ref supertraits, _) => (generics, supertraits),
            hir::ItemKind::TraitAlias(ref generics, ref supertraits) => (generics, supertraits),
            _ => span_bug!(item.span, "super_predicates invoked on non-trait"),
        };

        let icx = ItemCtxt::new(tcx, trait_def_id);

        // Convert the bounds that follow the colon, e.g., `Bar + Zed` in `trait Foo: Bar + Zed`.
        let self_param_ty = tcx.types.self_param;
        let superbounds1 = if let Some(assoc_name) = assoc_name {
            <dyn AstConv<'_>>::compute_bounds_that_match_assoc_type(
                &icx,
                self_param_ty,
                &bounds,
                assoc_name,
            )
        } else {
            <dyn AstConv<'_>>::compute_bounds(&icx, self_param_ty, &bounds)
        };

        let superbounds1 = superbounds1.predicates(tcx, self_param_ty);

        // Convert any explicit superbounds in the where-clause,
        // e.g., `trait Foo where Self: Bar`.
        // In the case of trait aliases, however, we include all bounds in the where-clause,
        // so e.g., `trait Foo = where u32: PartialEq<Self>` would include `u32: PartialEq<Self>`
        // as one of its "superpredicates".
        let is_trait_alias = tcx.is_trait_alias(trait_def_id);
        let superbounds2 = icx.type_parameter_bounds_in_generics(
            generics,
            item.hir_id(),
            self_param_ty,
            OnlySelfBounds(!is_trait_alias),
            assoc_name,
        );

        // Combine the two lists to form the complete set of superbounds:
        let superbounds = &*tcx.arena.alloc_from_iter(superbounds1.into_iter().chain(superbounds2));

        // Now require that immediate supertraits are converted,
        // which will, in turn, reach indirect supertraits.
        if assoc_name.is_none() {
            // Now require that immediate supertraits are converted,
            // which will, in turn, reach indirect supertraits.
            for &(pred, span) in superbounds {
                debug!("superbound: {:?}", pred);
                if let ty::PredicateKind::Trait(bound) = pred.kind().skip_binder() {
                    tcx.at(span).super_predicates_of(bound.def_id());
                }
            }
        }

        ty::GenericPredicates { parent: None, predicates: superbounds }
    } else {
        // if `assoc_name` is None, then the query should've been redirected to an
        // external provider
        assert!(assoc_name.is_some());
        tcx.super_predicates_of(trait_def_id)
    }
}

fn trait_def(tcx: TyCtxt<'_>, def_id: DefId) -> ty::TraitDef {
    let hir_id = tcx.hir().local_def_id_to_hir_id(def_id.expect_local());
    let item = tcx.hir().expect_item(hir_id);

    let (is_auto, unsafety) = match item.kind {
        hir::ItemKind::Trait(is_auto, unsafety, ..) => (is_auto == hir::IsAuto::Yes, unsafety),
        hir::ItemKind::TraitAlias(..) => (false, hir::Unsafety::Normal),
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
    let def_path_hash = tcx.def_path_hash(def_id);
    ty::TraitDef::new(
        def_id,
        unsafety,
        paren_sugar,
        is_auto,
        is_marker,
        skip_array_during_method_dispatch,
        spec_kind,
        def_path_hash,
    )
}

fn has_late_bound_regions<'tcx>(tcx: TyCtxt<'tcx>, node: Node<'tcx>) -> Option<Span> {
    struct LateBoundRegionsDetector<'tcx> {
        tcx: TyCtxt<'tcx>,
        outer_index: ty::DebruijnIndex,
        has_late_bound_regions: Option<Span>,
    }

    impl Visitor<'tcx> for LateBoundRegionsDetector<'tcx> {
        type Map = intravisit::ErasedMap<'tcx>;

        fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
            NestedVisitorMap::None
        }

        fn visit_ty(&mut self, ty: &'tcx hir::Ty<'tcx>) {
            if self.has_late_bound_regions.is_some() {
                return;
            }
            match ty.kind {
                hir::TyKind::BareFn(..) => {
                    self.outer_index.shift_in(1);
                    intravisit::walk_ty(self, ty);
                    self.outer_index.shift_out(1);
                }
                _ => intravisit::walk_ty(self, ty),
            }
        }

        fn visit_poly_trait_ref(
            &mut self,
            tr: &'tcx hir::PolyTraitRef<'tcx>,
            m: hir::TraitBoundModifier,
        ) {
            if self.has_late_bound_regions.is_some() {
                return;
            }
            self.outer_index.shift_in(1);
            intravisit::walk_poly_trait_ref(self, tr, m);
            self.outer_index.shift_out(1);
        }

        fn visit_lifetime(&mut self, lt: &'tcx hir::Lifetime) {
            if self.has_late_bound_regions.is_some() {
                return;
            }

            match self.tcx.named_region(lt.hir_id) {
                Some(rl::Region::Static | rl::Region::EarlyBound(..)) => {}
                Some(
                    rl::Region::LateBound(debruijn, _, _, _)
                    | rl::Region::LateBoundAnon(debruijn, _, _),
                ) if debruijn < self.outer_index => {}
                Some(
                    rl::Region::LateBound(..)
                    | rl::Region::LateBoundAnon(..)
                    | rl::Region::Free(..),
                )
                | None => {
                    self.has_late_bound_regions = Some(lt.span);
                }
            }
        }
    }

    fn has_late_bound_regions<'tcx>(
        tcx: TyCtxt<'tcx>,
        generics: &'tcx hir::Generics<'tcx>,
        decl: &'tcx hir::FnDecl<'tcx>,
    ) -> Option<Span> {
        let mut visitor = LateBoundRegionsDetector {
            tcx,
            outer_index: ty::INNERMOST,
            has_late_bound_regions: None,
        };
        for param in generics.params {
            if let GenericParamKind::Lifetime { .. } = param.kind {
                if tcx.is_late_bound(param.hir_id) {
                    return Some(param.span);
                }
            }
        }
        visitor.visit_fn_decl(decl);
        visitor.has_late_bound_regions
    }

    match node {
        Node::TraitItem(item) => match item.kind {
            hir::TraitItemKind::Fn(ref sig, _) => {
                has_late_bound_regions(tcx, &item.generics, &sig.decl)
            }
            _ => None,
        },
        Node::ImplItem(item) => match item.kind {
            hir::ImplItemKind::Fn(ref sig, _) => {
                has_late_bound_regions(tcx, &item.generics, &sig.decl)
            }
            _ => None,
        },
        Node::ForeignItem(item) => match item.kind {
            hir::ForeignItemKind::Fn(ref fn_decl, _, ref generics) => {
                has_late_bound_regions(tcx, generics, fn_decl)
            }
            _ => None,
        },
        Node::Item(item) => match item.kind {
            hir::ItemKind::Fn(ref sig, .., ref generics, _) => {
                has_late_bound_regions(tcx, generics, &sig.decl)
            }
            _ => None,
        },
        _ => None,
    }
}

struct AnonConstInParamTyDetector {
    in_param_ty: bool,
    found_anon_const_in_param_ty: bool,
    ct: HirId,
}

impl<'v> Visitor<'v> for AnonConstInParamTyDetector {
    type Map = intravisit::ErasedMap<'v>;

    fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
        NestedVisitorMap::None
    }

    fn visit_generic_param(&mut self, p: &'v hir::GenericParam<'v>) {
        if let GenericParamKind::Const { ref ty, default: _ } = p.kind {
            let prev = self.in_param_ty;
            self.in_param_ty = true;
            self.visit_ty(ty);
            self.in_param_ty = prev;
        }
    }

    fn visit_anon_const(&mut self, c: &'v hir::AnonConst) {
        if self.in_param_ty && self.ct == c.hir_id {
            self.found_anon_const_in_param_ty = true;
        } else {
            intravisit::walk_anon_const(self, c)
        }
    }
}

fn generics_of(tcx: TyCtxt<'_>, def_id: DefId) -> ty::Generics {
    use rustc_hir::*;

    let hir_id = tcx.hir().local_def_id_to_hir_id(def_id.expect_local());

    let node = tcx.hir().get(hir_id);
    let parent_def_id = match node {
        Node::ImplItem(_)
        | Node::TraitItem(_)
        | Node::Variant(_)
        | Node::Ctor(..)
        | Node::Field(_) => {
            let parent_id = tcx.hir().get_parent_item(hir_id);
            Some(tcx.hir().local_def_id(parent_id).to_def_id())
        }
        // FIXME(#43408) always enable this once `lazy_normalization` is
        // stable enough and does not need a feature gate anymore.
        Node::AnonConst(_) => {
            let parent_id = tcx.hir().get_parent_item(hir_id);
            let parent_def_id = tcx.hir().local_def_id(parent_id);

            let mut in_param_ty = false;
            for (_parent, node) in tcx.hir().parent_iter(hir_id) {
                if let Some(generics) = node.generics() {
                    let mut visitor = AnonConstInParamTyDetector {
                        in_param_ty: false,
                        found_anon_const_in_param_ty: false,
                        ct: hir_id,
                    };

                    visitor.visit_generics(generics);
                    in_param_ty = visitor.found_anon_const_in_param_ty;
                    break;
                }
            }

            if in_param_ty {
                // We do not allow generic parameters in anon consts if we are inside
                // of a const parameter type, e.g. `struct Foo<const N: usize, const M: [u8; N]>` is not allowed.
                None
            } else if tcx.lazy_normalization() {
                if let Some(param_id) = tcx.hir().opt_const_param_default_param_hir_id(hir_id) {
                    // If the def_id we are calling generics_of on is an anon ct default i.e:
                    //
                    // struct Foo<const N: usize = { .. }>;
                    //        ^^^       ^          ^^^^^^ def id of this anon const
                    //        ^         ^ param_id
                    //        ^ parent_def_id
                    //
                    // then we only want to return generics for params to the left of `N`. If we don't do that we
                    // end up with that const looking like: `ty::ConstKind::Unevaluated(def_id, substs: [N#0])`.
                    //
                    // This causes ICEs (#86580) when building the substs for Foo in `fn foo() -> Foo { .. }` as
                    // we substitute the defaults with the partially built substs when we build the substs. Subst'ing
                    // the `N#0` on the unevaluated const indexes into the empty substs we're in the process of building.
                    //
                    // We fix this by having this function return the parent's generics ourselves and truncating the
                    // generics to only include non-forward declared params (with the exception of the `Self` ty)
                    //
                    // For the above code example that means we want `substs: []`
                    // For the following struct def we want `substs: [N#0]` when generics_of is called on
                    // the def id of the `{ N + 1 }` anon const
                    // struct Foo<const N: usize, const M: usize = { N + 1 }>;
                    //
                    // This has some implications for how we get the predicates available to the anon const
                    // see `explicit_predicates_of` for more information on this
                    let generics = tcx.generics_of(parent_def_id.to_def_id());
                    let param_def = tcx.hir().local_def_id(param_id).to_def_id();
                    let param_def_idx = generics.param_def_id_to_index[&param_def];
                    // In the above example this would be .params[..N#0]
                    let params = generics.params[..param_def_idx as usize].to_owned();
                    let param_def_id_to_index =
                        params.iter().map(|param| (param.def_id, param.index)).collect();

                    return ty::Generics {
                        // we set the parent of these generics to be our parent's parent so that we
                        // dont end up with substs: [N, M, N] for the const default on a struct like this:
                        // struct Foo<const N: usize, const M: usize = { ... }>;
                        parent: generics.parent,
                        parent_count: generics.parent_count,
                        params,
                        param_def_id_to_index,
                        has_self: generics.has_self,
                        has_late_bound_regions: generics.has_late_bound_regions,
                    };
                }

                // HACK(eddyb) this provides the correct generics when
                // `feature(generic_const_expressions)` is enabled, so that const expressions
                // used with const generics, e.g. `Foo<{N+1}>`, can work at all.
                //
                // Note that we do not supply the parent generics when using
                // `min_const_generics`.
                Some(parent_def_id.to_def_id())
            } else {
                let parent_node = tcx.hir().get(tcx.hir().get_parent_node(hir_id));
                match parent_node {
                    // HACK(eddyb) this provides the correct generics for repeat
                    // expressions' count (i.e. `N` in `[x; N]`), and explicit
                    // `enum` discriminants (i.e. `D` in `enum Foo { Bar = D }`),
                    // as they shouldn't be able to cause query cycle errors.
                    Node::Expr(&Expr { kind: ExprKind::Repeat(_, ref constant), .. })
                    | Node::Variant(Variant { disr_expr: Some(ref constant), .. })
                        if constant.hir_id == hir_id =>
                    {
                        Some(parent_def_id.to_def_id())
                    }

                    _ => None,
                }
            }
        }
        Node::Expr(&hir::Expr { kind: hir::ExprKind::Closure(..), .. }) => {
            Some(tcx.closure_base_def_id(def_id))
        }
        Node::Item(item) => match item.kind {
            ItemKind::OpaqueTy(hir::OpaqueTy { impl_trait_fn, .. }) => {
                impl_trait_fn.or_else(|| {
                    let parent_id = tcx.hir().get_parent_item(hir_id);
                    assert!(parent_id != hir_id && parent_id != CRATE_HIR_ID);
                    debug!("generics_of: parent of opaque ty {:?} is {:?}", def_id, parent_id);
                    // Opaque types are always nested within another item, and
                    // inherit the generics of the item.
                    Some(tcx.hir().local_def_id(parent_id).to_def_id())
                })
            }
            _ => None,
        },
        _ => None,
    };

    let mut opt_self = None;
    let mut allow_defaults = false;

    let no_generics = hir::Generics::empty();
    let ast_generics = match node {
        Node::TraitItem(item) => &item.generics,

        Node::ImplItem(item) => &item.generics,

        Node::Item(item) => {
            match item.kind {
                ItemKind::Fn(.., ref generics, _)
                | ItemKind::Impl(hir::Impl { ref generics, .. }) => generics,

                ItemKind::TyAlias(_, ref generics)
                | ItemKind::Enum(_, ref generics)
                | ItemKind::Struct(_, ref generics)
                | ItemKind::OpaqueTy(hir::OpaqueTy { ref generics, .. })
                | ItemKind::Union(_, ref generics) => {
                    allow_defaults = true;
                    generics
                }

                ItemKind::Trait(_, _, ref generics, ..)
                | ItemKind::TraitAlias(ref generics, ..) => {
                    // Add in the self type parameter.
                    //
                    // Something of a hack: use the node id for the trait, also as
                    // the node id for the Self type parameter.
                    let param_id = item.def_id;

                    opt_self = Some(ty::GenericParamDef {
                        index: 0,
                        name: kw::SelfUpper,
                        def_id: param_id.to_def_id(),
                        pure_wrt_drop: false,
                        kind: ty::GenericParamDefKind::Type {
                            has_default: false,
                            object_lifetime_default: rl::Set1::Empty,
                            synthetic: None,
                        },
                    });

                    allow_defaults = true;
                    generics
                }

                _ => &no_generics,
            }
        }

        Node::ForeignItem(item) => match item.kind {
            ForeignItemKind::Static(..) => &no_generics,
            ForeignItemKind::Fn(_, _, ref generics) => generics,
            ForeignItemKind::Type => &no_generics,
        },

        _ => &no_generics,
    };

    let has_self = opt_self.is_some();
    let mut parent_has_self = false;
    let mut own_start = has_self as u32;
    let parent_count = parent_def_id.map_or(0, |def_id| {
        let generics = tcx.generics_of(def_id);
        assert_eq!(has_self, false);
        parent_has_self = generics.has_self;
        own_start = generics.count() as u32;
        generics.parent_count + generics.params.len()
    });

    let mut params: Vec<_> = Vec::with_capacity(ast_generics.params.len() + has_self as usize);

    if let Some(opt_self) = opt_self {
        params.push(opt_self);
    }

    let early_lifetimes = early_bound_lifetimes_from_generics(tcx, ast_generics);
    params.extend(early_lifetimes.enumerate().map(|(i, param)| ty::GenericParamDef {
        name: param.name.ident().name,
        index: own_start + i as u32,
        def_id: tcx.hir().local_def_id(param.hir_id).to_def_id(),
        pure_wrt_drop: param.pure_wrt_drop,
        kind: ty::GenericParamDefKind::Lifetime,
    }));

    let object_lifetime_defaults = tcx.object_lifetime_defaults(hir_id);

    // Now create the real type and const parameters.
    let type_start = own_start - has_self as u32 + params.len() as u32;
    let mut i = 0;

    params.extend(ast_generics.params.iter().filter_map(|param| match param.kind {
        GenericParamKind::Lifetime { .. } => None,
        GenericParamKind::Type { ref default, synthetic, .. } => {
            if !allow_defaults && default.is_some() {
                if !tcx.features().default_type_parameter_fallback {
                    tcx.struct_span_lint_hir(
                        lint::builtin::INVALID_TYPE_PARAM_DEFAULT,
                        param.hir_id,
                        param.span,
                        |lint| {
                            lint.build(
                                "defaults for type parameters are only allowed in \
                                 `struct`, `enum`, `type`, or `trait` definitions",
                            )
                            .emit();
                        },
                    );
                }
            }

            let kind = ty::GenericParamDefKind::Type {
                has_default: default.is_some(),
                object_lifetime_default: object_lifetime_defaults
                    .as_ref()
                    .map_or(rl::Set1::Empty, |o| o[i]),
                synthetic,
            };

            let param_def = ty::GenericParamDef {
                index: type_start + i as u32,
                name: param.name.ident().name,
                def_id: tcx.hir().local_def_id(param.hir_id).to_def_id(),
                pure_wrt_drop: param.pure_wrt_drop,
                kind,
            };
            i += 1;
            Some(param_def)
        }
        GenericParamKind::Const { default, .. } => {
            if !allow_defaults && default.is_some() {
                tcx.sess.span_err(
                    param.span,
                    "defaults for const parameters are only allowed in \
                    `struct`, `enum`, `type`, or `trait` definitions",
                );
            }

            let param_def = ty::GenericParamDef {
                index: type_start + i as u32,
                name: param.name.ident().name,
                def_id: tcx.hir().local_def_id(param.hir_id).to_def_id(),
                pure_wrt_drop: param.pure_wrt_drop,
                kind: ty::GenericParamDefKind::Const { has_default: default.is_some() },
            };
            i += 1;
            Some(param_def)
        }
    }));

    // provide junk type parameter defs - the only place that
    // cares about anything but the length is instantiation,
    // and we don't do that for closures.
    if let Node::Expr(&hir::Expr { kind: hir::ExprKind::Closure(.., gen), .. }) = node {
        let dummy_args = if gen.is_some() {
            &["<resume_ty>", "<yield_ty>", "<return_ty>", "<witness>", "<upvars>"][..]
        } else {
            &["<closure_kind>", "<closure_signature>", "<upvars>"][..]
        };

        params.extend(dummy_args.iter().enumerate().map(|(i, &arg)| ty::GenericParamDef {
            index: type_start + i as u32,
            name: Symbol::intern(arg),
            def_id,
            pure_wrt_drop: false,
            kind: ty::GenericParamDefKind::Type {
                has_default: false,
                object_lifetime_default: rl::Set1::Empty,
                synthetic: None,
            },
        }));
    }

    let param_def_id_to_index = params.iter().map(|param| (param.def_id, param.index)).collect();

    ty::Generics {
        parent: parent_def_id,
        parent_count,
        params,
        param_def_id_to_index,
        has_self: has_self || parent_has_self,
        has_late_bound_regions: has_late_bound_regions(tcx, node),
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
    use hir::TyKind::*;
    match &ty.kind {
        Infer => true,
        Slice(ty) | Array(ty, _) => is_suggestable_infer_ty(ty),
        Tup(tys) => tys.iter().any(is_suggestable_infer_ty),
        Ptr(mut_ty) | Rptr(_, mut_ty) => is_suggestable_infer_ty(mut_ty.ty),
        OpaqueDef(_, generic_args) => are_suggestable_generic_args(generic_args),
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

pub fn get_infer_ret_ty(output: &'hir hir::FnRetTy<'hir>) -> Option<&'hir hir::Ty<'hir>> {
    if let hir::FnRetTy::Return(ref ty) = output {
        if is_suggestable_infer_ty(ty) {
            return Some(&**ty);
        }
    }
    None
}

fn fn_sig(tcx: TyCtxt<'_>, def_id: DefId) -> ty::PolyFnSig<'_> {
    use rustc_hir::Node::*;
    use rustc_hir::*;

    let def_id = def_id.expect_local();
    let hir_id = tcx.hir().local_def_id_to_hir_id(def_id);

    let icx = ItemCtxt::new(tcx, def_id.to_def_id());

    match tcx.hir().get(hir_id) {
        TraitItem(hir::TraitItem {
            kind: TraitItemKind::Fn(sig, TraitFn::Provided(_)),
            ident,
            generics,
            ..
        })
        | ImplItem(hir::ImplItem { kind: ImplItemKind::Fn(sig, _), ident, generics, .. })
        | Item(hir::Item { kind: ItemKind::Fn(sig, generics, _), ident, .. }) => {
            match get_infer_ret_ty(&sig.decl.output) {
                Some(ty) => {
                    let fn_sig = tcx.typeck(def_id).liberated_fn_sigs()[hir_id];
                    // Typeck doesn't expect erased regions to be returned from `type_of`.
                    let fn_sig = tcx.fold_regions(fn_sig, &mut false, |r, _| match r {
                        ty::ReErased => tcx.lifetimes.re_static,
                        _ => r,
                    });
                    let fn_sig = ty::Binder::dummy(fn_sig);

                    let mut visitor = PlaceholderHirTyCollector::default();
                    visitor.visit_ty(ty);
                    let mut diag = bad_placeholder_type(tcx, visitor.0, "return type");
                    let ret_ty = fn_sig.skip_binder().output();
                    if ret_ty != tcx.ty_error() {
                        if !ret_ty.is_closure() {
                            let ret_ty_str = match ret_ty.kind() {
                                // Suggest a function pointer return type instead of a unique function definition
                                // (e.g. `fn() -> i32` instead of `fn() -> i32 { f }`, the latter of which is invalid
                                // syntax)
                                ty::FnDef(..) => ret_ty.fn_sig(tcx).to_string(),
                                _ => ret_ty.to_string(),
                            };
                            diag.span_suggestion(
                                ty.span,
                                "replace with the correct return type",
                                ret_ty_str,
                                Applicability::MaybeIncorrect,
                            );
                        } else {
                            // We're dealing with a closure, so we should suggest using `impl Fn` or trait bounds
                            // to prevent the user from getting a papercut while trying to use the unique closure
                            // syntax (e.g. `[closure@src/lib.rs:2:5: 2:9]`).
                            diag.help("consider using an `Fn`, `FnMut`, or `FnOnce` trait bound");
                            diag.note("for more information on `Fn` traits and closure types, see https://doc.rust-lang.org/book/ch13-01-closures.html");
                        }
                    }
                    diag.emit();

                    fn_sig
                }
                None => <dyn AstConv<'_>>::ty_of_fn(
                    &icx,
                    hir_id,
                    sig.header.unsafety,
                    sig.header.abi,
                    &sig.decl,
                    &generics,
                    Some(ident.span),
                    None,
                ),
            }
        }

        TraitItem(hir::TraitItem {
            kind: TraitItemKind::Fn(FnSig { header, decl, span: _ }, _),
            ident,
            generics,
            ..
        }) => <dyn AstConv<'_>>::ty_of_fn(
            &icx,
            hir_id,
            header.unsafety,
            header.abi,
            decl,
            &generics,
            Some(ident.span),
            None,
        ),

        ForeignItem(&hir::ForeignItem {
            kind: ForeignItemKind::Fn(ref fn_decl, _, _),
            ident,
            ..
        }) => {
            let abi = tcx.hir().get_foreign_abi(hir_id);
            compute_sig_of_foreign_fn_decl(tcx, def_id.to_def_id(), fn_decl, abi, ident)
        }

        Ctor(data) | Variant(hir::Variant { data, .. }) if data.ctor_hir_id().is_some() => {
            let ty = tcx.type_of(tcx.hir().get_parent_did(hir_id).to_def_id());
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

        Expr(&hir::Expr { kind: hir::ExprKind::Closure(..), .. }) => {
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

fn impl_trait_ref(tcx: TyCtxt<'_>, def_id: DefId) -> Option<ty::TraitRef<'_>> {
    let icx = ItemCtxt::new(tcx, def_id);

    let hir_id = tcx.hir().local_def_id_to_hir_id(def_id.expect_local());
    match tcx.hir().expect_item(hir_id).kind {
        hir::ItemKind::Impl(ref impl_) => impl_.of_trait.as_ref().map(|ast_trait_ref| {
            let selfty = tcx.type_of(def_id);
            <dyn AstConv<'_>>::instantiate_mono_trait_ref(&icx, ast_trait_ref, selfty)
        }),
        _ => bug!(),
    }
}

fn impl_polarity(tcx: TyCtxt<'_>, def_id: DefId) -> ty::ImplPolarity {
    let hir_id = tcx.hir().local_def_id_to_hir_id(def_id.expect_local());
    let is_rustc_reservation = tcx.has_attr(def_id, sym::rustc_reservation_impl);
    let item = tcx.hir().expect_item(hir_id);
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
fn predicates_defined_on(tcx: TyCtxt<'_>, def_id: DefId) -> ty::GenericPredicates<'_> {
    debug!("predicates_defined_on({:?})", def_id);
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

/// Returns a list of all type predicates (explicit and implicit) for the definition with
/// ID `def_id`. This includes all predicates returned by `predicates_defined_on`, plus
/// `Self: Trait` predicates for traits.
fn predicates_of(tcx: TyCtxt<'_>, def_id: DefId) -> ty::GenericPredicates<'_> {
    let mut result = tcx.predicates_defined_on(def_id);

    if tcx.is_trait(def_id) {
        // For traits, add `Self: Trait` predicate. This is
        // not part of the predicates that a user writes, but it
        // is something that one must prove in order to invoke a
        // method or project an associated type.
        //
        // In the chalk setup, this predicate is not part of the
        // "predicates" for a trait item. But it is useful in
        // rustc because if you directly (e.g.) invoke a trait
        // method like `Trait::method(...)`, you must naturally
        // prove that the trait applies to the types that were
        // used, and adding the predicate into this list ensures
        // that this is done.
        let mut span = tcx.def_span(def_id);
        if tcx.sess.source_map().is_local_span(span) {
            // `guess_head_span` reads the actual source file from
            // disk to try to determine the 'head' snippet of the span.
            // Don't do this for a span that comes from a file outside
            // of our crate, since this would make our query output
            // (and overall crate metadata) dependent on the
            // *current* state of an external file.
            span = tcx.sess.source_map().guess_head_span(span);
        }
        result.predicates =
            tcx.arena.alloc_from_iter(result.predicates.iter().copied().chain(std::iter::once((
                ty::TraitRef::identity(tcx, def_id).without_const().to_predicate(tcx),
                span,
            ))));
    }
    debug!("predicates_of(def_id={:?}) = {:?}", def_id, result);
    result
}

/// Returns a list of user-specified type predicates for the definition with ID `def_id`.
/// N.B., this does not include any implied/inferred constraints.
fn gather_explicit_predicates_of(tcx: TyCtxt<'_>, def_id: DefId) -> ty::GenericPredicates<'_> {
    use rustc_hir::*;

    debug!("explicit_predicates_of(def_id={:?})", def_id);

    let hir_id = tcx.hir().local_def_id_to_hir_id(def_id.expect_local());
    let node = tcx.hir().get(hir_id);

    let mut is_trait = None;
    let mut is_default_impl_trait = None;

    let icx = ItemCtxt::new(tcx, def_id);

    const NO_GENERICS: &hir::Generics<'_> = &hir::Generics::empty();

    // We use an `IndexSet` to preserves order of insertion.
    // Preserving the order of insertion is important here so as not to break UI tests.
    let mut predicates: FxIndexSet<(ty::Predicate<'_>, Span)> = FxIndexSet::default();

    let ast_generics = match node {
        Node::TraitItem(item) => &item.generics,

        Node::ImplItem(item) => &item.generics,

        Node::Item(item) => {
            match item.kind {
                ItemKind::Impl(ref impl_) => {
                    if impl_.defaultness.is_default() {
                        is_default_impl_trait = tcx
                            .impl_trait_ref(def_id)
                            .map(|trait_ref| ty::Binder::dummy(trait_ref));
                    }
                    &impl_.generics
                }
                ItemKind::Fn(.., ref generics, _)
                | ItemKind::TyAlias(_, ref generics)
                | ItemKind::Enum(_, ref generics)
                | ItemKind::Struct(_, ref generics)
                | ItemKind::Union(_, ref generics) => generics,

                ItemKind::Trait(_, _, ref generics, ..) => {
                    is_trait = Some(ty::TraitRef::identity(tcx, def_id));
                    generics
                }
                ItemKind::TraitAlias(ref generics, _) => {
                    is_trait = Some(ty::TraitRef::identity(tcx, def_id));
                    generics
                }
                ItemKind::OpaqueTy(OpaqueTy {
                    bounds: _,
                    impl_trait_fn,
                    ref generics,
                    origin: _,
                }) => {
                    if impl_trait_fn.is_some() {
                        // return-position impl trait
                        //
                        // We don't inherit predicates from the parent here:
                        // If we have, say `fn f<'a, T: 'a>() -> impl Sized {}`
                        // then the return type is `f::<'static, T>::{{opaque}}`.
                        //
                        // If we inherited the predicates of `f` then we would
                        // require that `T: 'static` to show that the return
                        // type is well-formed.
                        //
                        // The only way to have something with this opaque type
                        // is from the return type of the containing function,
                        // which will ensure that the function's predicates
                        // hold.
                        return ty::GenericPredicates { parent: None, predicates: &[] };
                    } else {
                        // type-alias impl trait
                        generics
                    }
                }

                _ => NO_GENERICS,
            }
        }

        Node::ForeignItem(item) => match item.kind {
            ForeignItemKind::Static(..) => NO_GENERICS,
            ForeignItemKind::Fn(_, _, ref generics) => generics,
            ForeignItemKind::Type => NO_GENERICS,
        },

        _ => NO_GENERICS,
    };

    let generics = tcx.generics_of(def_id);
    let parent_count = generics.parent_count as u32;
    let has_own_self = generics.has_self && parent_count == 0;

    // Below we'll consider the bounds on the type parameters (including `Self`)
    // and the explicit where-clauses, but to get the full set of predicates
    // on a trait we need to add in the supertrait bounds and bounds found on
    // associated types.
    if let Some(_trait_ref) = is_trait {
        predicates.extend(tcx.super_predicates_of(def_id).predicates.iter().cloned());
    }

    // In default impls, we can assume that the self type implements
    // the trait. So in:
    //
    //     default impl Foo for Bar { .. }
    //
    // we add a default where clause `Foo: Bar`. We do a similar thing for traits
    // (see below). Recall that a default impl is not itself an impl, but rather a
    // set of defaults that can be incorporated into another impl.
    if let Some(trait_ref) = is_default_impl_trait {
        predicates.insert((trait_ref.without_const().to_predicate(tcx), tcx.def_span(def_id)));
    }

    // Collect the region predicates that were declared inline as
    // well. In the case of parameters declared on a fn or method, we
    // have to be careful to only iterate over early-bound regions.
    let mut index = parent_count + has_own_self as u32;
    for param in early_bound_lifetimes_from_generics(tcx, ast_generics) {
        let region = tcx.mk_region(ty::ReEarlyBound(ty::EarlyBoundRegion {
            def_id: tcx.hir().local_def_id(param.hir_id).to_def_id(),
            index,
            name: param.name.ident().name,
        }));
        index += 1;

        match param.kind {
            GenericParamKind::Lifetime { .. } => {
                param.bounds.iter().for_each(|bound| match bound {
                    hir::GenericBound::Outlives(lt) => {
                        let bound = <dyn AstConv<'_>>::ast_region_to_region(&icx, &lt, None);
                        let outlives = ty::Binder::dummy(ty::OutlivesPredicate(region, bound));
                        predicates.insert((outlives.to_predicate(tcx), lt.span));
                    }
                    _ => bug!(),
                });
            }
            _ => bug!(),
        }
    }

    // Collect the predicates that were written inline by the user on each
    // type parameter (e.g., `<T: Foo>`).
    for param in ast_generics.params {
        match param.kind {
            // We already dealt with early bound lifetimes above.
            GenericParamKind::Lifetime { .. } => (),
            GenericParamKind::Type { .. } => {
                let name = param.name.ident().name;
                let param_ty = ty::ParamTy::new(index, name).to_ty(tcx);
                index += 1;

                let mut bounds = <dyn AstConv<'_>>::compute_bounds(&icx, param_ty, &param.bounds);
                // Params are implicitly sized unless a `?Sized` bound is found
                <dyn AstConv<'_>>::add_implicitly_sized(
                    &icx,
                    &mut bounds,
                    &param.bounds,
                    Some((param.hir_id, ast_generics.where_clause.predicates)),
                    param.span,
                );
                predicates.extend(bounds.predicates(tcx, param_ty));
            }
            GenericParamKind::Const { .. } => {
                // Bounds on const parameters are currently not possible.
                debug_assert!(param.bounds.is_empty());
                index += 1;
            }
        }
    }

    // Add in the bounds that appear in the where-clause.
    let where_clause = &ast_generics.where_clause;
    for predicate in where_clause.predicates {
        match predicate {
            hir::WherePredicate::BoundPredicate(bound_pred) => {
                let ty = icx.to_ty(&bound_pred.bounded_ty);
                let bound_vars = icx.tcx.late_bound_vars(bound_pred.bounded_ty.hir_id);

                // Keep the type around in a dummy predicate, in case of no bounds.
                // That way, `where Ty:` is not a complete noop (see #53696) and `Ty`
                // is still checked for WF.
                if bound_pred.bounds.is_empty() {
                    if let ty::Param(_) = ty.kind() {
                        // This is a `where T:`, which can be in the HIR from the
                        // transformation that moves `?Sized` to `T`'s declaration.
                        // We can skip the predicate because type parameters are
                        // trivially WF, but also we *should*, to avoid exposing
                        // users who never wrote `where Type:,` themselves, to
                        // compiler/tooling bugs from not handling WF predicates.
                    } else {
                        let span = bound_pred.bounded_ty.span;
                        let re_root_empty = tcx.lifetimes.re_root_empty;
                        let predicate = ty::Binder::bind_with_vars(
                            ty::PredicateKind::TypeOutlives(ty::OutlivesPredicate(
                                ty,
                                re_root_empty,
                            )),
                            bound_vars,
                        );
                        predicates.insert((predicate.to_predicate(tcx), span));
                    }
                }

                let mut bounds = Bounds::default();
                <dyn AstConv<'_>>::add_bounds(
                    &icx,
                    ty,
                    bound_pred.bounds.iter(),
                    &mut bounds,
                    bound_vars,
                );
                predicates.extend(bounds.predicates(tcx, ty));
            }

            hir::WherePredicate::RegionPredicate(region_pred) => {
                let r1 = <dyn AstConv<'_>>::ast_region_to_region(&icx, &region_pred.lifetime, None);
                predicates.extend(region_pred.bounds.iter().map(|bound| {
                    let (r2, span) = match bound {
                        hir::GenericBound::Outlives(lt) => {
                            (<dyn AstConv<'_>>::ast_region_to_region(&icx, lt, None), lt.span)
                        }
                        _ => bug!(),
                    };
                    let pred = ty::Binder::dummy(ty::PredicateKind::RegionOutlives(
                        ty::OutlivesPredicate(r1, r2),
                    ))
                    .to_predicate(icx.tcx);

                    (pred, span)
                }))
            }

            hir::WherePredicate::EqPredicate(..) => {
                // FIXME(#20041)
            }
        }
    }

    if tcx.features().generic_const_exprs {
        predicates.extend(const_evaluatable_predicates_of(tcx, def_id.expect_local()));
    }

    let mut predicates: Vec<_> = predicates.into_iter().collect();

    // Subtle: before we store the predicates into the tcx, we
    // sort them so that predicates like `T: Foo<Item=U>` come
    // before uses of `U`.  This avoids false ambiguity errors
    // in trait checking. See `setup_constraining_predicates`
    // for details.
    if let Node::Item(&Item { kind: ItemKind::Impl { .. }, .. }) = node {
        let self_ty = tcx.type_of(def_id);
        let trait_ref = tcx.impl_trait_ref(def_id);
        cgp::setup_constraining_predicates(
            tcx,
            &mut predicates,
            trait_ref,
            &mut cgp::parameters_for_impl(tcx, self_ty, trait_ref),
        );
    }

    let result = ty::GenericPredicates {
        parent: generics.parent,
        predicates: tcx.arena.alloc_from_iter(predicates),
    };
    debug!("explicit_predicates_of(def_id={:?}) = {:?}", def_id, result);
    result
}

fn const_evaluatable_predicates_of<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: LocalDefId,
) -> FxIndexSet<(ty::Predicate<'tcx>, Span)> {
    struct ConstCollector<'tcx> {
        tcx: TyCtxt<'tcx>,
        preds: FxIndexSet<(ty::Predicate<'tcx>, Span)>,
    }

    impl<'tcx> intravisit::Visitor<'tcx> for ConstCollector<'tcx> {
        type Map = Map<'tcx>;

        fn nested_visit_map(&mut self) -> intravisit::NestedVisitorMap<Self::Map> {
            intravisit::NestedVisitorMap::None
        }

        fn visit_anon_const(&mut self, c: &'tcx hir::AnonConst) {
            let def_id = self.tcx.hir().local_def_id(c.hir_id);
            let ct = ty::Const::from_anon_const(self.tcx, def_id);
            if let ty::ConstKind::Unevaluated(uv) = ct.val {
                assert_eq!(uv.promoted, None);
                let span = self.tcx.hir().span(c.hir_id);
                self.preds.insert((
                    ty::Binder::dummy(ty::PredicateKind::ConstEvaluatable(uv.shrink()))
                        .to_predicate(self.tcx),
                    span,
                ));
            }
        }

        fn visit_const_param_default(&mut self, _param: HirId, _ct: &'tcx hir::AnonConst) {
            // Do not look into const param defaults,
            // these get checked when they are actually instantiated.
            //
            // We do not want the following to error:
            //
            //     struct Foo<const N: usize, const M: usize = { N + 1 }>;
            //     struct Bar<const N: usize>(Foo<N, 3>);
        }
    }

    let hir_id = tcx.hir().local_def_id_to_hir_id(def_id);
    let node = tcx.hir().get(hir_id);

    let mut collector = ConstCollector { tcx, preds: FxIndexSet::default() };
    if let hir::Node::Item(item) = node {
        if let hir::ItemKind::Impl(ref impl_) = item.kind {
            if let Some(of_trait) = &impl_.of_trait {
                debug!("const_evaluatable_predicates_of({:?}): visit impl trait_ref", def_id);
                collector.visit_trait_ref(of_trait);
            }

            debug!("const_evaluatable_predicates_of({:?}): visit_self_ty", def_id);
            collector.visit_ty(impl_.self_ty);
        }
    }

    if let Some(generics) = node.generics() {
        debug!("const_evaluatable_predicates_of({:?}): visit_generics", def_id);
        collector.visit_generics(generics);
    }

    if let Some(fn_sig) = tcx.hir().fn_sig_by_hir_id(hir_id) {
        debug!("const_evaluatable_predicates_of({:?}): visit_fn_decl", def_id);
        collector.visit_fn_decl(fn_sig.decl);
    }
    debug!("const_evaluatable_predicates_of({:?}) = {:?}", def_id, collector.preds);

    collector.preds
}

fn trait_explicit_predicates_and_bounds(
    tcx: TyCtxt<'_>,
    def_id: LocalDefId,
) -> ty::GenericPredicates<'_> {
    assert_eq!(tcx.def_kind(def_id), DefKind::Trait);
    gather_explicit_predicates_of(tcx, def_id.to_def_id())
}

fn explicit_predicates_of(tcx: TyCtxt<'_>, def_id: DefId) -> ty::GenericPredicates<'_> {
    let def_kind = tcx.def_kind(def_id);
    if let DefKind::Trait = def_kind {
        // Remove bounds on associated types from the predicates, they will be
        // returned by `explicit_item_bounds`.
        let predicates_and_bounds = tcx.trait_explicit_predicates_and_bounds(def_id.expect_local());
        let trait_identity_substs = InternalSubsts::identity_for_item(tcx, def_id);

        let is_assoc_item_ty = |ty: Ty<'_>| {
            // For a predicate from a where clause to become a bound on an
            // associated type:
            // * It must use the identity substs of the item.
            //     * Since any generic parameters on the item are not in scope,
            //       this means that the item is not a GAT, and its identity
            //       substs are the same as the trait's.
            // * It must be an associated type for this trait (*not* a
            //   supertrait).
            if let ty::Projection(projection) = ty.kind() {
                projection.substs == trait_identity_substs
                    && tcx.associated_item(projection.item_def_id).container.id() == def_id
            } else {
                false
            }
        };

        let predicates: Vec<_> = predicates_and_bounds
            .predicates
            .iter()
            .copied()
            .filter(|(pred, _)| match pred.kind().skip_binder() {
                ty::PredicateKind::Trait(tr) => !is_assoc_item_ty(tr.self_ty()),
                ty::PredicateKind::Projection(proj) => {
                    !is_assoc_item_ty(proj.projection_ty.self_ty())
                }
                ty::PredicateKind::TypeOutlives(outlives) => !is_assoc_item_ty(outlives.0),
                _ => true,
            })
            .collect();
        if predicates.len() == predicates_and_bounds.predicates.len() {
            predicates_and_bounds
        } else {
            ty::GenericPredicates {
                parent: predicates_and_bounds.parent,
                predicates: tcx.arena.alloc_slice(&predicates),
            }
        }
    } else {
        if matches!(def_kind, DefKind::AnonConst) && tcx.lazy_normalization() {
            let hir_id = tcx.hir().local_def_id_to_hir_id(def_id.expect_local());
            if let Some(_) = tcx.hir().opt_const_param_default_param_hir_id(hir_id) {
                // In `generics_of` we set the generics' parent to be our parent's parent which means that
                // we lose out on the predicates of our actual parent if we dont return those predicates here.
                // (See comment in `generics_of` for more information on why the parent shenanigans is necessary)
                //
                // struct Foo<T, const N: usize = { <T as Trait>::ASSOC }>(T) where T: Trait;
                //        ^^^                     ^^^^^^^^^^^^^^^^^^^^^^^ the def id we are calling
                //        ^^^                                             explicit_predicates_of on
                //        parent item we dont have set as the
                //        parent of generics returned by `generics_of`
                //
                // In the above code we want the anon const to have predicates in its param env for `T: Trait`
                let item_id = tcx.hir().get_parent_item(hir_id);
                let item_def_id = tcx.hir().local_def_id(item_id).to_def_id();
                // In the above code example we would be calling `explicit_predicates_of(Foo)` here
                return tcx.explicit_predicates_of(item_def_id);
            }
        }
        gather_explicit_predicates_of(tcx, def_id)
    }
}

/// Converts a specific `GenericBound` from the AST into a set of
/// predicates that apply to the self type. A vector is returned
/// because this can be anywhere from zero predicates (`T: ?Sized` adds no
/// predicates) to one (`T: Foo`) to many (`T: Bar<X = i32>` adds `T: Bar`
/// and `<T as Bar>::X == i32`).
fn predicates_from_bound<'tcx>(
    astconv: &dyn AstConv<'tcx>,
    param_ty: Ty<'tcx>,
    bound: &'tcx hir::GenericBound<'tcx>,
) -> Vec<(ty::Predicate<'tcx>, Span)> {
    let mut bounds = Bounds::default();
    astconv.add_bounds(
        param_ty,
        std::array::IntoIter::new([bound]),
        &mut bounds,
        ty::List::empty(),
    );
    bounds.predicates(astconv.tcx(), param_ty)
}

fn compute_sig_of_foreign_fn_decl<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: DefId,
    decl: &'tcx hir::FnDecl<'tcx>,
    abi: abi::Abi,
    ident: Ident,
) -> ty::PolyFnSig<'tcx> {
    let unsafety = if abi == abi::Abi::RustIntrinsic {
        intrinsic_operation_unsafety(tcx.item_name(def_id))
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
        &hir::Generics::empty(),
        Some(ident.span),
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
            check(&input, ty)
        }
        if let hir::FnRetTy::Return(ref ty) = decl.output {
            check(&ty, fty.output().skip_binder())
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

fn static_mutability(tcx: TyCtxt<'_>, def_id: DefId) -> Option<hir::Mutability> {
    match tcx.hir().get_if_local(def_id) {
        Some(
            Node::Item(&hir::Item { kind: hir::ItemKind::Static(_, mutbl, _), .. })
            | Node::ForeignItem(&hir::ForeignItem {
                kind: hir::ForeignItemKind::Static(_, mutbl),
                ..
            }),
        ) => Some(mutbl),
        Some(_) => None,
        _ => bug!("static_mutability applied to non-local def-id {:?}", def_id),
    }
}

fn generator_kind(tcx: TyCtxt<'_>, def_id: DefId) -> Option<hir::GeneratorKind> {
    match tcx.hir().get_if_local(def_id) {
        Some(Node::Expr(&rustc_hir::Expr {
            kind: rustc_hir::ExprKind::Closure(_, _, body_id, _, _),
            ..
        })) => tcx.hir().body(body_id).generator_kind(),
        Some(_) => None,
        _ => bug!("generator_kind applied to non-local def-id {:?}", def_id),
    }
}

fn from_target_feature(
    tcx: TyCtxt<'_>,
    id: DefId,
    attr: &ast::Attribute,
    supported_target_features: &FxHashMap<String, Option<Symbol>>,
    target_features: &mut Vec<Symbol>,
) {
    let list = match attr.meta_item_list() {
        Some(list) => list,
        None => return,
    };
    let bad_item = |span| {
        let msg = "malformed `target_feature` attribute input";
        let code = "enable = \"..\"".to_owned();
        tcx.sess
            .struct_span_err(span, &msg)
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
        let value = match item.value_str() {
            Some(value) => value,
            None => {
                bad_item(item.span());
                continue;
            }
        };

        // We allow comma separation to enable multiple features.
        target_features.extend(value.as_str().split(',').filter_map(|feature| {
            let feature_gate = match supported_target_features.get(feature) {
                Some(g) => g,
                None => {
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
                }
            };

            // Only allow features whose feature gates have been enabled.
            let allowed = match feature_gate.as_ref().copied() {
                Some(sym::arm_target_feature) => rust_features.arm_target_feature,
                Some(sym::aarch64_target_feature) => rust_features.aarch64_target_feature,
                Some(sym::hexagon_target_feature) => rust_features.hexagon_target_feature,
                Some(sym::powerpc_target_feature) => rust_features.powerpc_target_feature,
                Some(sym::mips_target_feature) => rust_features.mips_target_feature,
                Some(sym::riscv_target_feature) => rust_features.riscv_target_feature,
                Some(sym::avx512_target_feature) => rust_features.avx512_target_feature,
                Some(sym::sse4a_target_feature) => rust_features.sse4a_target_feature,
                Some(sym::tbm_target_feature) => rust_features.tbm_target_feature,
                Some(sym::wasm_target_feature) => rust_features.wasm_target_feature,
                Some(sym::cmpxchg16b_target_feature) => rust_features.cmpxchg16b_target_feature,
                Some(sym::adx_target_feature) => rust_features.adx_target_feature,
                Some(sym::movbe_target_feature) => rust_features.movbe_target_feature,
                Some(sym::rtm_target_feature) => rust_features.rtm_target_feature,
                Some(sym::f16c_target_feature) => rust_features.f16c_target_feature,
                Some(sym::ermsb_target_feature) => rust_features.ermsb_target_feature,
                Some(sym::bpf_target_feature) => rust_features.bpf_target_feature,
                Some(name) => bug!("unknown target feature gate {}", name),
                None => true,
            };
            if !allowed && id.is_local() {
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

fn linkage_by_name(tcx: TyCtxt<'_>, def_id: DefId, name: &str) -> Linkage {
    use rustc_middle::mir::mono::Linkage::*;

    // Use the names from src/llvm/docs/LangRef.rst here. Most types are only
    // applicable to variable declarations and may not really make sense for
    // Rust code in the first place but allow them anyway and trust that the
    // user knows what s/he's doing. Who knows, unanticipated use cases may pop
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
        _ => {
            let span = tcx.hir().span_if_local(def_id);
            if let Some(span) = span {
                tcx.sess.span_fatal(span, "invalid linkage specified")
            } else {
                tcx.sess.fatal(&format!("invalid linkage specified: {}", name))
            }
        }
    }
}

fn codegen_fn_attrs(tcx: TyCtxt<'_>, id: DefId) -> CodegenFnAttrs {
    let attrs = tcx.get_attrs(id);

    let mut codegen_fn_attrs = CodegenFnAttrs::new();
    if tcx.should_inherit_track_caller(id) {
        codegen_fn_attrs.flags |= CodegenFnAttrFlags::TRACK_CALLER;
    }

    // With -Z panic-in-drop=abort, drop_in_place never unwinds.
    if tcx.sess.opts.debugging_opts.panic_in_drop == PanicStrategy::Abort {
        if Some(id) == tcx.lang_items().drop_in_place_fn() {
            codegen_fn_attrs.flags |= CodegenFnAttrFlags::NEVER_UNWIND;
        }
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
            if tcx.is_foreign_item(id) {
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
            if tcx.is_foreign_item(id) {
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
            if tcx.is_foreign_item(id) {
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
        } else if attr.has_name(sym::naked) {
            codegen_fn_attrs.flags |= CodegenFnAttrFlags::NAKED;
        } else if attr.has_name(sym::no_mangle) {
            codegen_fn_attrs.flags |= CodegenFnAttrFlags::NO_MANGLE;
        } else if attr.has_name(sym::no_coverage) {
            codegen_fn_attrs.flags |= CodegenFnAttrFlags::NO_COVERAGE;
        } else if attr.has_name(sym::rustc_std_internal_symbol) {
            codegen_fn_attrs.flags |= CodegenFnAttrFlags::RUSTC_STD_INTERNAL_SYMBOL;
        } else if attr.has_name(sym::used) {
            codegen_fn_attrs.flags |= CodegenFnAttrFlags::USED;
        } else if attr.has_name(sym::cmse_nonsecure_entry) {
            if !matches!(tcx.fn_sig(id).abi(), abi::Abi::C { .. }) {
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
            if !tcx.is_closure(id) && tcx.fn_sig(id).abi() != abi::Abi::Rust {
                struct_span_err!(tcx.sess, attr.span, E0737, "`#[track_caller]` requires Rust ABI")
                    .emit();
            }
            if tcx.is_closure(id) && !tcx.features().closure_track_caller {
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
            if !tcx.is_closure(id) && tcx.fn_sig(id).unsafety() == hir::Unsafety::Normal {
                if tcx.sess.target.is_like_wasm || tcx.sess.opts.actually_rustdoc {
                    // The `#[target_feature]` attribute is allowed on
                    // WebAssembly targets on all functions, including safe
                    // ones. Other targets require that `#[target_feature]` is
                    // only applied to unsafe funtions (pending the
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
                    err.span_label(tcx.def_span(id), "not an `unsafe` function");
                    err.emit();
                } else if let Some(local_id) = id.as_local() {
                    check_target_feature_trait_unsafe(tcx, local_id, attr.span);
                }
            }
            from_target_feature(
                tcx,
                id,
                attr,
                &supported_target_features,
                &mut codegen_fn_attrs.target_features,
            );
        } else if attr.has_name(sym::linkage) {
            if let Some(val) = attr.value_str() {
                codegen_fn_attrs.linkage = Some(linkage_by_name(tcx, id, &val.as_str()));
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
                    } else if item.has_name(sym::memory) {
                        codegen_fn_attrs.no_sanitize |= SanitizerSet::MEMORY;
                    } else if item.has_name(sym::thread) {
                        codegen_fn_attrs.no_sanitize |= SanitizerSet::THREAD;
                    } else if item.has_name(sym::hwaddress) {
                        codegen_fn_attrs.no_sanitize |= SanitizerSet::HWADDRESS;
                    } else {
                        tcx.sess
                            .struct_span_err(item.span(), "invalid argument for `no_sanitize`")
                            .note("expected one of: `address`, `hwaddress`, `memory` or `thread`")
                            .emit();
                    }
                }
            }
        } else if attr.has_name(sym::instruction_set) {
            codegen_fn_attrs.instruction_set = match attr.meta().map(|i| i.kind) {
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
        match attr.meta().map(|i| i.kind) {
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
                } else if list_contains_name(&items[..], sym::always) {
                    InlineAttr::Always
                } else if list_contains_name(&items[..], sym::never) {
                    InlineAttr::Never
                } else {
                    struct_span_err!(
                        tcx.sess.diagnostic(),
                        items[0].span(),
                        E0535,
                        "invalid argument"
                    )
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
        match attr.meta().map(|i| i.kind) {
            Some(MetaItemKind::Word) => {
                err(attr.span, "expected one argument");
                ia
            }
            Some(MetaItemKind::List(ref items)) => {
                inline_span = Some(attr.span);
                if items.len() != 1 {
                    err(attr.span, "expected one argument");
                    OptimizeAttr::None
                } else if list_contains_name(&items[..], sym::size) {
                    OptimizeAttr::Size
                } else if list_contains_name(&items[..], sym::speed) {
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
    if tcx.features().target_feature_11 && tcx.is_closure(id) {
        let owner_id = tcx.parent(id).expect("closure should have a parent");
        codegen_fn_attrs
            .target_features
            .extend(tcx.codegen_fn_attrs(owner_id).target_features.iter().copied())
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
                let hir_id = tcx.hir().local_def_id_to_hir_id(id.expect_local());
                tcx.struct_span_lint_hir(
                    lint::builtin::INLINE_NO_SANITIZE,
                    hir_id,
                    no_sanitize_span,
                    |lint| {
                        lint.build("`no_sanitize` will have no effect after inlining")
                            .span_note(inline_span, "inlining requested here")
                            .emit();
                    },
                )
            }
        }
    }

    // Weak lang items have the same semantics as "std internal" symbols in the
    // sense that they're preserved through all our LTO passes and only
    // strippable by the linker.
    //
    // Additionally weak lang items have predetermined symbol names.
    if tcx.is_weak_lang_item(id) {
        codegen_fn_attrs.flags |= CodegenFnAttrFlags::RUSTC_STD_INTERNAL_SYMBOL;
    }
    let check_name = |attr: &Attribute, sym| attr.has_name(sym);
    if let Some(name) = weak_lang_items::link_name(check_name, &attrs) {
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

/// Checks if the provided DefId is a method in a trait impl for a trait which has track_caller
/// applied to the method prototype.
fn should_inherit_track_caller(tcx: TyCtxt<'_>, def_id: DefId) -> bool {
    if let Some(impl_item) = tcx.opt_associated_item(def_id) {
        if let ty::AssocItemContainer::ImplContainer(impl_def_id) = impl_item.container {
            if let Some(trait_def_id) = tcx.trait_id_of_impl(impl_def_id) {
                if let Some(trait_item) = tcx
                    .associated_items(trait_def_id)
                    .filter_by_name_unhygienic(impl_item.ident.name)
                    .find(move |trait_item| {
                        trait_item.kind == ty::AssocKind::Fn
                            && tcx.hygienic_eq(impl_item.ident, trait_item.ident, trait_def_id)
                    })
                {
                    return tcx
                        .codegen_fn_attrs(trait_item.def_id)
                        .flags
                        .intersects(CodegenFnAttrFlags::TRACK_CALLER);
                }
            }
        }
    }

    false
}

fn check_link_ordinal(tcx: TyCtxt<'_>, attr: &ast::Attribute) -> Option<usize> {
    use rustc_ast::{Lit, LitIntType, LitKind};
    let meta_item_list = attr.meta_item_list();
    let meta_item_list: Option<&[ast::NestedMetaItem]> = meta_item_list.as_ref().map(Vec::as_ref);
    let sole_meta_list = match meta_item_list {
        Some([item]) => item.literal(),
        _ => None,
    };
    if let Some(Lit { kind: LitKind::Int(ordinal, LitIntType::Unsuffixed), .. }) = sole_meta_list {
        if *ordinal <= usize::MAX as u128 {
            Some(*ordinal as usize)
        } else {
            let msg = format!("ordinal value in `link_ordinal` is too large: `{}`", &ordinal);
            tcx.sess
                .struct_span_err(attr.span, &msg)
                .note("the value may not exceed `usize::MAX`")
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
        let parent_item = tcx.hir().expect_item(parent_id);
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
