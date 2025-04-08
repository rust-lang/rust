//! This module defines the primary IR[^1] used in rustdoc together with the procedures that
//! transform rustc data types into it.
//!
//! This IR — commonly referred to as the *cleaned AST* — is modeled after the [AST][ast].
//!
//! There are two kinds of transformation — *cleaning* — procedures:
//!
//! 1. Cleans [HIR][hir] types. Used for user-written code and inlined local re-exports
//!    both found in the local crate.
//! 2. Cleans [`rustc_middle::ty`] types. Used for inlined cross-crate re-exports and anything
//!    output by the trait solver (e.g., when synthesizing blanket and auto-trait impls).
//!    They usually have `ty` or `middle` in their name.
//!
//! Their name is prefixed by `clean_`.
//!
//! Both the HIR and the `rustc_middle::ty` IR are quite removed from the source code.
//! The cleaned AST on the other hand is closer to it which simplifies the rendering process.
//! Furthermore, operating on a single IR instead of two avoids duplicating efforts down the line.
//!
//! This IR is consumed by both the HTML and the JSON backend.
//!
//! [^1]: Intermediate representation.

mod auto_trait;
mod blanket_impl;
pub(crate) mod cfg;
pub(crate) mod inline;
mod render_macro_matchers;
mod simplify;
pub(crate) mod types;
pub(crate) mod utils;

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::mem;

use rustc_ast::token::{Token, TokenKind};
use rustc_ast::tokenstream::{TokenStream, TokenTree};
use rustc_data_structures::fx::{FxHashMap, FxHashSet, FxIndexMap, FxIndexSet, IndexEntry};
use rustc_errors::codes::*;
use rustc_errors::{FatalError, struct_span_code_err};
use rustc_hir::PredicateOrigin;
use rustc_hir::def::{CtorKind, DefKind, Res};
use rustc_hir::def_id::{DefId, DefIdMap, DefIdSet, LOCAL_CRATE, LocalDefId};
use rustc_hir_analysis::hir_ty_lowering::FeedConstTy;
use rustc_hir_analysis::{lower_const_arg_for_rustdoc, lower_ty};
use rustc_middle::metadata::Reexport;
use rustc_middle::middle::resolve_bound_vars as rbv;
use rustc_middle::ty::{self, AdtKind, GenericArgsRef, Ty, TyCtxt, TypeVisitableExt, TypingMode};
use rustc_middle::{bug, span_bug};
use rustc_span::ExpnKind;
use rustc_span::hygiene::{AstPass, MacroKind};
use rustc_span::symbol::{Ident, Symbol, kw, sym};
use rustc_trait_selection::traits::wf::object_region_bounds;
use thin_vec::ThinVec;
use tracing::{debug, instrument};
use utils::*;
use {rustc_ast as ast, rustc_hir as hir};

pub(crate) use self::types::*;
pub(crate) use self::utils::{krate, register_res, synthesize_auto_trait_and_blanket_impls};
use crate::core::DocContext;
use crate::formats::item_type::ItemType;
use crate::visit_ast::Module as DocModule;

pub(crate) fn clean_doc_module<'tcx>(doc: &DocModule<'tcx>, cx: &mut DocContext<'tcx>) -> Item {
    let mut items: Vec<Item> = vec![];
    let mut inserted = FxHashSet::default();
    items.extend(doc.foreigns.iter().map(|(item, renamed)| {
        let item = clean_maybe_renamed_foreign_item(cx, item, *renamed);
        if let Some(name) = item.name
            && (cx.render_options.document_hidden || !item.is_doc_hidden())
        {
            inserted.insert((item.type_(), name));
        }
        item
    }));
    items.extend(doc.mods.iter().filter_map(|x| {
        if !inserted.insert((ItemType::Module, x.name)) {
            return None;
        }
        let item = clean_doc_module(x, cx);
        if !cx.render_options.document_hidden && item.is_doc_hidden() {
            // Hidden modules are stripped at a later stage.
            // If a hidden module has the same name as a visible one, we want
            // to keep both of them around.
            inserted.remove(&(ItemType::Module, x.name));
        }
        Some(item)
    }));

    // Split up imports from all other items.
    //
    // This covers the case where somebody does an import which should pull in an item,
    // but there's already an item with the same namespace and same name. Rust gives
    // priority to the not-imported one, so we should, too.
    items.extend(doc.items.values().flat_map(|(item, renamed, import_id)| {
        // First, lower everything other than glob imports.
        if matches!(item.kind, hir::ItemKind::Use(_, hir::UseKind::Glob)) {
            return Vec::new();
        }
        let v = clean_maybe_renamed_item(cx, item, *renamed, *import_id);
        for item in &v {
            if let Some(name) = item.name
                && (cx.render_options.document_hidden || !item.is_doc_hidden())
            {
                inserted.insert((item.type_(), name));
            }
        }
        v
    }));
    items.extend(doc.inlined_foreigns.iter().flat_map(|((_, renamed), (res, local_import_id))| {
        let Some(def_id) = res.opt_def_id() else { return Vec::new() };
        let name = renamed.unwrap_or_else(|| cx.tcx.item_name(def_id));
        let import = cx.tcx.hir_expect_item(*local_import_id);
        match import.kind {
            hir::ItemKind::Use(path, kind) => {
                let hir::UsePath { segments, span, .. } = *path;
                let path = hir::Path { segments, res: *res, span };
                clean_use_statement_inner(import, name, &path, kind, cx, &mut Default::default())
            }
            _ => unreachable!(),
        }
    }));
    items.extend(doc.items.values().flat_map(|(item, renamed, _)| {
        // Now we actually lower the imports, skipping everything else.
        if let hir::ItemKind::Use(path, hir::UseKind::Glob) = item.kind {
            let name = renamed.unwrap_or(kw::Empty); // using kw::Empty is a bit of a hack
            clean_use_statement(item, name, path, hir::UseKind::Glob, cx, &mut inserted)
        } else {
            // skip everything else
            Vec::new()
        }
    }));

    // determine if we should display the inner contents or
    // the outer `mod` item for the source code.

    let span = Span::new({
        let where_outer = doc.where_outer(cx.tcx);
        let sm = cx.sess().source_map();
        let outer = sm.lookup_char_pos(where_outer.lo());
        let inner = sm.lookup_char_pos(doc.where_inner.lo());
        if outer.file.start_pos == inner.file.start_pos {
            // mod foo { ... }
            where_outer
        } else {
            // mod foo; (and a separate SourceFile for the contents)
            doc.where_inner
        }
    });

    let kind = ModuleItem(Module { items, span });
    generate_item_with_correct_attrs(
        cx,
        kind,
        doc.def_id.to_def_id(),
        doc.name,
        doc.import_id,
        doc.renamed,
    )
}

fn is_glob_import(tcx: TyCtxt<'_>, import_id: LocalDefId) -> bool {
    if let hir::Node::Item(item) = tcx.hir_node_by_def_id(import_id)
        && let hir::ItemKind::Use(_, use_kind) = item.kind
    {
        use_kind == hir::UseKind::Glob
    } else {
        false
    }
}

fn generate_item_with_correct_attrs(
    cx: &mut DocContext<'_>,
    kind: ItemKind,
    def_id: DefId,
    name: Symbol,
    import_id: Option<LocalDefId>,
    renamed: Option<Symbol>,
) -> Item {
    let target_attrs = inline::load_attrs(cx, def_id);
    let attrs = if let Some(import_id) = import_id {
        // glob reexports are treated the same as `#[doc(inline)]` items.
        //
        // For glob re-exports the item may or may not exist to be re-exported (potentially the cfgs
        // on the path up until the glob can be removed, and only cfgs on the globbed item itself
        // matter), for non-inlined re-exports see #85043.
        let is_inline = hir_attr_lists(inline::load_attrs(cx, import_id.to_def_id()), sym::doc)
            .get_word_attr(sym::inline)
            .is_some()
            || (is_glob_import(cx.tcx, import_id)
                && (cx.render_options.document_hidden || !cx.tcx.is_doc_hidden(def_id)));
        let mut attrs = get_all_import_attributes(cx, import_id, def_id, is_inline);
        add_without_unwanted_attributes(&mut attrs, target_attrs, is_inline, None);
        attrs
    } else {
        // We only keep the item's attributes.
        target_attrs.iter().map(|attr| (Cow::Borrowed(attr), None)).collect()
    };
    let cfg = extract_cfg_from_attrs(
        attrs.iter().map(move |(attr, _)| match attr {
            Cow::Borrowed(attr) => *attr,
            Cow::Owned(attr) => attr,
        }),
        cx.tcx,
        &cx.cache.hidden_cfg,
    );
    let attrs = Attributes::from_hir_iter(attrs.iter().map(|(attr, did)| (&**attr, *did)), false);

    let name = renamed.or(Some(name));
    let mut item = Item::from_def_id_and_attrs_and_parts(def_id, name, kind, attrs, cfg);
    item.inner.inline_stmt_id = import_id;
    item
}

fn clean_generic_bound<'tcx>(
    bound: &hir::GenericBound<'tcx>,
    cx: &mut DocContext<'tcx>,
) -> Option<GenericBound> {
    Some(match *bound {
        hir::GenericBound::Outlives(lt) => GenericBound::Outlives(clean_lifetime(lt, cx)),
        hir::GenericBound::Trait(ref t) => {
            // `T: ~const Destruct` is hidden because `T: Destruct` is a no-op.
            if let hir::BoundConstness::Maybe(_) = t.modifiers.constness
                && cx.tcx.lang_items().destruct_trait() == Some(t.trait_ref.trait_def_id().unwrap())
            {
                return None;
            }

            GenericBound::TraitBound(clean_poly_trait_ref(t, cx), t.modifiers)
        }
        hir::GenericBound::Use(args, ..) => {
            GenericBound::Use(args.iter().map(|arg| clean_precise_capturing_arg(arg, cx)).collect())
        }
    })
}

pub(crate) fn clean_trait_ref_with_constraints<'tcx>(
    cx: &mut DocContext<'tcx>,
    trait_ref: ty::PolyTraitRef<'tcx>,
    constraints: ThinVec<AssocItemConstraint>,
) -> Path {
    let kind = cx.tcx.def_kind(trait_ref.def_id()).into();
    if !matches!(kind, ItemType::Trait | ItemType::TraitAlias) {
        span_bug!(cx.tcx.def_span(trait_ref.def_id()), "`TraitRef` had unexpected kind {kind:?}");
    }
    inline::record_extern_fqn(cx, trait_ref.def_id(), kind);
    let path = clean_middle_path(
        cx,
        trait_ref.def_id(),
        true,
        constraints,
        trait_ref.map_bound(|tr| tr.args),
    );

    debug!(?trait_ref);

    path
}

fn clean_poly_trait_ref_with_constraints<'tcx>(
    cx: &mut DocContext<'tcx>,
    poly_trait_ref: ty::PolyTraitRef<'tcx>,
    constraints: ThinVec<AssocItemConstraint>,
) -> GenericBound {
    GenericBound::TraitBound(
        PolyTrait {
            trait_: clean_trait_ref_with_constraints(cx, poly_trait_ref, constraints),
            generic_params: clean_bound_vars(poly_trait_ref.bound_vars()),
        },
        hir::TraitBoundModifiers::NONE,
    )
}

fn clean_lifetime(lifetime: &hir::Lifetime, cx: &DocContext<'_>) -> Lifetime {
    if let Some(
        rbv::ResolvedArg::EarlyBound(did)
        | rbv::ResolvedArg::LateBound(_, _, did)
        | rbv::ResolvedArg::Free(_, did),
    ) = cx.tcx.named_bound_var(lifetime.hir_id)
        && let Some(lt) = cx.args.get(&did.to_def_id()).and_then(|arg| arg.as_lt())
    {
        return *lt;
    }
    Lifetime(lifetime.ident.name)
}

pub(crate) fn clean_precise_capturing_arg(
    arg: &hir::PreciseCapturingArg<'_>,
    cx: &DocContext<'_>,
) -> PreciseCapturingArg {
    match arg {
        hir::PreciseCapturingArg::Lifetime(lt) => {
            PreciseCapturingArg::Lifetime(clean_lifetime(lt, cx))
        }
        hir::PreciseCapturingArg::Param(param) => PreciseCapturingArg::Param(param.ident.name),
    }
}

pub(crate) fn clean_const<'tcx>(
    constant: &hir::ConstArg<'tcx>,
    _cx: &mut DocContext<'tcx>,
) -> ConstantKind {
    match &constant.kind {
        hir::ConstArgKind::Path(qpath) => {
            ConstantKind::Path { path: qpath_to_string(qpath).into() }
        }
        hir::ConstArgKind::Anon(anon) => ConstantKind::Anonymous { body: anon.body },
        hir::ConstArgKind::Infer(..) => ConstantKind::Infer,
    }
}

pub(crate) fn clean_middle_const<'tcx>(
    constant: ty::Binder<'tcx, ty::Const<'tcx>>,
    _cx: &mut DocContext<'tcx>,
) -> ConstantKind {
    // FIXME: instead of storing the stringified expression, store `self` directly instead.
    ConstantKind::TyConst { expr: constant.skip_binder().to_string().into() }
}

pub(crate) fn clean_middle_region(region: ty::Region<'_>) -> Option<Lifetime> {
    match region.kind() {
        ty::ReStatic => Some(Lifetime::statik()),
        _ if !region.has_name() => None,
        ty::ReBound(_, ty::BoundRegion { kind: ty::BoundRegionKind::Named(_, name), .. }) => {
            Some(Lifetime(name))
        }
        ty::ReEarlyParam(ref data) => Some(Lifetime(data.name)),
        ty::ReBound(..)
        | ty::ReLateParam(..)
        | ty::ReVar(..)
        | ty::ReError(_)
        | ty::RePlaceholder(..)
        | ty::ReErased => {
            debug!("cannot clean region {region:?}");
            None
        }
    }
}

fn clean_where_predicate<'tcx>(
    predicate: &hir::WherePredicate<'tcx>,
    cx: &mut DocContext<'tcx>,
) -> Option<WherePredicate> {
    if !predicate.kind.in_where_clause() {
        return None;
    }
    Some(match *predicate.kind {
        hir::WherePredicateKind::BoundPredicate(ref wbp) => {
            let bound_params = wbp
                .bound_generic_params
                .iter()
                .map(|param| clean_generic_param(cx, None, param))
                .collect();
            WherePredicate::BoundPredicate {
                ty: clean_ty(wbp.bounded_ty, cx),
                bounds: wbp.bounds.iter().filter_map(|x| clean_generic_bound(x, cx)).collect(),
                bound_params,
            }
        }

        hir::WherePredicateKind::RegionPredicate(ref wrp) => WherePredicate::RegionPredicate {
            lifetime: clean_lifetime(wrp.lifetime, cx),
            bounds: wrp.bounds.iter().filter_map(|x| clean_generic_bound(x, cx)).collect(),
        },

        hir::WherePredicateKind::EqPredicate(ref wrp) => WherePredicate::EqPredicate {
            lhs: clean_ty(wrp.lhs_ty, cx),
            rhs: clean_ty(wrp.rhs_ty, cx).into(),
        },
    })
}

pub(crate) fn clean_predicate<'tcx>(
    predicate: ty::Clause<'tcx>,
    cx: &mut DocContext<'tcx>,
) -> Option<WherePredicate> {
    let bound_predicate = predicate.kind();
    match bound_predicate.skip_binder() {
        ty::ClauseKind::Trait(pred) => clean_poly_trait_predicate(bound_predicate.rebind(pred), cx),
        ty::ClauseKind::RegionOutlives(pred) => Some(clean_region_outlives_predicate(pred)),
        ty::ClauseKind::TypeOutlives(pred) => {
            Some(clean_type_outlives_predicate(bound_predicate.rebind(pred), cx))
        }
        ty::ClauseKind::Projection(pred) => {
            Some(clean_projection_predicate(bound_predicate.rebind(pred), cx))
        }
        // FIXME(generic_const_exprs): should this do something?
        ty::ClauseKind::ConstEvaluatable(..)
        | ty::ClauseKind::WellFormed(..)
        | ty::ClauseKind::ConstArgHasType(..)
        // FIXME(const_trait_impl): We can probably use this `HostEffect` pred to render `~const`.
        | ty::ClauseKind::HostEffect(_) => None,
    }
}

fn clean_poly_trait_predicate<'tcx>(
    pred: ty::PolyTraitPredicate<'tcx>,
    cx: &mut DocContext<'tcx>,
) -> Option<WherePredicate> {
    // `T: ~const Destruct` is hidden because `T: Destruct` is a no-op.
    // FIXME(const_trait_impl) check constness
    if Some(pred.skip_binder().def_id()) == cx.tcx.lang_items().destruct_trait() {
        return None;
    }

    let poly_trait_ref = pred.map_bound(|pred| pred.trait_ref);
    Some(WherePredicate::BoundPredicate {
        ty: clean_middle_ty(poly_trait_ref.self_ty(), cx, None, None),
        bounds: vec![clean_poly_trait_ref_with_constraints(cx, poly_trait_ref, ThinVec::new())],
        bound_params: Vec::new(),
    })
}

fn clean_region_outlives_predicate(pred: ty::RegionOutlivesPredicate<'_>) -> WherePredicate {
    let ty::OutlivesPredicate(a, b) = pred;

    WherePredicate::RegionPredicate {
        lifetime: clean_middle_region(a).expect("failed to clean lifetime"),
        bounds: vec![GenericBound::Outlives(
            clean_middle_region(b).expect("failed to clean bounds"),
        )],
    }
}

fn clean_type_outlives_predicate<'tcx>(
    pred: ty::Binder<'tcx, ty::TypeOutlivesPredicate<'tcx>>,
    cx: &mut DocContext<'tcx>,
) -> WherePredicate {
    let ty::OutlivesPredicate(ty, lt) = pred.skip_binder();

    WherePredicate::BoundPredicate {
        ty: clean_middle_ty(pred.rebind(ty), cx, None, None),
        bounds: vec![GenericBound::Outlives(
            clean_middle_region(lt).expect("failed to clean lifetimes"),
        )],
        bound_params: Vec::new(),
    }
}

fn clean_middle_term<'tcx>(
    term: ty::Binder<'tcx, ty::Term<'tcx>>,
    cx: &mut DocContext<'tcx>,
) -> Term {
    match term.skip_binder().unpack() {
        ty::TermKind::Ty(ty) => Term::Type(clean_middle_ty(term.rebind(ty), cx, None, None)),
        ty::TermKind::Const(c) => Term::Constant(clean_middle_const(term.rebind(c), cx)),
    }
}

fn clean_hir_term<'tcx>(term: &hir::Term<'tcx>, cx: &mut DocContext<'tcx>) -> Term {
    match term {
        hir::Term::Ty(ty) => Term::Type(clean_ty(ty, cx)),
        hir::Term::Const(c) => {
            let ct = lower_const_arg_for_rustdoc(cx.tcx, c, FeedConstTy::No);
            Term::Constant(clean_middle_const(ty::Binder::dummy(ct), cx))
        }
    }
}

fn clean_projection_predicate<'tcx>(
    pred: ty::Binder<'tcx, ty::ProjectionPredicate<'tcx>>,
    cx: &mut DocContext<'tcx>,
) -> WherePredicate {
    WherePredicate::EqPredicate {
        lhs: clean_projection(
            pred.map_bound(|p| {
                // FIXME: This needs to be made resilient for `AliasTerm`s that
                // are associated consts.
                p.projection_term.expect_ty(cx.tcx)
            }),
            cx,
            None,
        ),
        rhs: clean_middle_term(pred.map_bound(|p| p.term), cx),
    }
}

fn clean_projection<'tcx>(
    ty: ty::Binder<'tcx, ty::AliasTy<'tcx>>,
    cx: &mut DocContext<'tcx>,
    def_id: Option<DefId>,
) -> Type {
    if cx.tcx.is_impl_trait_in_trait(ty.skip_binder().def_id) {
        return clean_middle_opaque_bounds(cx, ty.skip_binder().def_id, ty.skip_binder().args);
    }

    let trait_ = clean_trait_ref_with_constraints(
        cx,
        ty.map_bound(|ty| ty.trait_ref(cx.tcx)),
        ThinVec::new(),
    );
    let self_type = clean_middle_ty(ty.map_bound(|ty| ty.self_ty()), cx, None, None);
    let self_def_id = if let Some(def_id) = def_id {
        cx.tcx.opt_parent(def_id).or(Some(def_id))
    } else {
        self_type.def_id(&cx.cache)
    };
    let should_show_cast = compute_should_show_cast(self_def_id, &trait_, &self_type);
    Type::QPath(Box::new(QPathData {
        assoc: projection_to_path_segment(ty, cx),
        should_show_cast,
        self_type,
        trait_: Some(trait_),
    }))
}

fn compute_should_show_cast(self_def_id: Option<DefId>, trait_: &Path, self_type: &Type) -> bool {
    !trait_.segments.is_empty()
        && self_def_id
            .zip(Some(trait_.def_id()))
            .map_or(!self_type.is_self_type(), |(id, trait_)| id != trait_)
}

fn projection_to_path_segment<'tcx>(
    ty: ty::Binder<'tcx, ty::AliasTy<'tcx>>,
    cx: &mut DocContext<'tcx>,
) -> PathSegment {
    let def_id = ty.skip_binder().def_id;
    let item = cx.tcx.associated_item(def_id);
    let generics = cx.tcx.generics_of(def_id);
    PathSegment {
        name: item.name,
        args: GenericArgs::AngleBracketed {
            args: clean_middle_generic_args(
                cx,
                ty.map_bound(|ty| &ty.args[generics.parent_count..]),
                false,
                def_id,
            ),
            constraints: Default::default(),
        },
    }
}

fn clean_generic_param_def(
    def: &ty::GenericParamDef,
    defaults: ParamDefaults,
    cx: &mut DocContext<'_>,
) -> GenericParamDef {
    let (name, kind) = match def.kind {
        ty::GenericParamDefKind::Lifetime => {
            (def.name, GenericParamDefKind::Lifetime { outlives: ThinVec::new() })
        }
        ty::GenericParamDefKind::Type { has_default, synthetic, .. } => {
            let default = if let ParamDefaults::Yes = defaults
                && has_default
            {
                Some(clean_middle_ty(
                    ty::Binder::dummy(cx.tcx.type_of(def.def_id).instantiate_identity()),
                    cx,
                    Some(def.def_id),
                    None,
                ))
            } else {
                None
            };
            (
                def.name,
                GenericParamDefKind::Type {
                    bounds: ThinVec::new(), // These are filled in from the where-clauses.
                    default: default.map(Box::new),
                    synthetic,
                },
            )
        }
        ty::GenericParamDefKind::Const { has_default, synthetic } => (
            def.name,
            GenericParamDefKind::Const {
                ty: Box::new(clean_middle_ty(
                    ty::Binder::dummy(
                        cx.tcx
                            .type_of(def.def_id)
                            .no_bound_vars()
                            .expect("const parameter types cannot be generic"),
                    ),
                    cx,
                    Some(def.def_id),
                    None,
                )),
                default: if let ParamDefaults::Yes = defaults
                    && has_default
                {
                    Some(Box::new(
                        cx.tcx.const_param_default(def.def_id).instantiate_identity().to_string(),
                    ))
                } else {
                    None
                },
                synthetic,
            },
        ),
    };

    GenericParamDef { name, def_id: def.def_id, kind }
}

/// Whether to clean generic parameter defaults or not.
enum ParamDefaults {
    Yes,
    No,
}

fn clean_generic_param<'tcx>(
    cx: &mut DocContext<'tcx>,
    generics: Option<&hir::Generics<'tcx>>,
    param: &hir::GenericParam<'tcx>,
) -> GenericParamDef {
    let (name, kind) = match param.kind {
        hir::GenericParamKind::Lifetime { .. } => {
            let outlives = if let Some(generics) = generics {
                generics
                    .outlives_for_param(param.def_id)
                    .filter(|bp| !bp.in_where_clause)
                    .flat_map(|bp| bp.bounds)
                    .map(|bound| match bound {
                        hir::GenericBound::Outlives(lt) => clean_lifetime(lt, cx),
                        _ => panic!(),
                    })
                    .collect()
            } else {
                ThinVec::new()
            };
            (param.name.ident().name, GenericParamDefKind::Lifetime { outlives })
        }
        hir::GenericParamKind::Type { ref default, synthetic } => {
            let bounds = if let Some(generics) = generics {
                generics
                    .bounds_for_param(param.def_id)
                    .filter(|bp| bp.origin != PredicateOrigin::WhereClause)
                    .flat_map(|bp| bp.bounds)
                    .filter_map(|x| clean_generic_bound(x, cx))
                    .collect()
            } else {
                ThinVec::new()
            };
            (
                param.name.ident().name,
                GenericParamDefKind::Type {
                    bounds,
                    default: default.map(|t| clean_ty(t, cx)).map(Box::new),
                    synthetic,
                },
            )
        }
        hir::GenericParamKind::Const { ty, default, synthetic } => (
            param.name.ident().name,
            GenericParamDefKind::Const {
                ty: Box::new(clean_ty(ty, cx)),
                default: default.map(|ct| {
                    Box::new(lower_const_arg_for_rustdoc(cx.tcx, ct, FeedConstTy::No).to_string())
                }),
                synthetic,
            },
        ),
    };

    GenericParamDef { name, def_id: param.def_id.to_def_id(), kind }
}

/// Synthetic type-parameters are inserted after normal ones.
/// In order for normal parameters to be able to refer to synthetic ones,
/// scans them first.
fn is_impl_trait(param: &hir::GenericParam<'_>) -> bool {
    match param.kind {
        hir::GenericParamKind::Type { synthetic, .. } => synthetic,
        _ => false,
    }
}

/// This can happen for `async fn`, e.g. `async fn f<'_>(&'_ self)`.
///
/// See `lifetime_to_generic_param` in `rustc_ast_lowering` for more information.
fn is_elided_lifetime(param: &hir::GenericParam<'_>) -> bool {
    matches!(
        param.kind,
        hir::GenericParamKind::Lifetime { kind: hir::LifetimeParamKind::Elided(_) }
    )
}

pub(crate) fn clean_generics<'tcx>(
    gens: &hir::Generics<'tcx>,
    cx: &mut DocContext<'tcx>,
) -> Generics {
    let impl_trait_params = gens
        .params
        .iter()
        .filter(|param| is_impl_trait(param))
        .map(|param| {
            let param = clean_generic_param(cx, Some(gens), param);
            match param.kind {
                GenericParamDefKind::Lifetime { .. } => unreachable!(),
                GenericParamDefKind::Type { ref bounds, .. } => {
                    cx.impl_trait_bounds.insert(param.def_id.into(), bounds.to_vec());
                }
                GenericParamDefKind::Const { .. } => unreachable!(),
            }
            param
        })
        .collect::<Vec<_>>();

    let mut bound_predicates = FxIndexMap::default();
    let mut region_predicates = FxIndexMap::default();
    let mut eq_predicates = ThinVec::default();
    for pred in gens.predicates.iter().filter_map(|x| clean_where_predicate(x, cx)) {
        match pred {
            WherePredicate::BoundPredicate { ty, bounds, bound_params } => {
                match bound_predicates.entry(ty) {
                    IndexEntry::Vacant(v) => {
                        v.insert((bounds, bound_params));
                    }
                    IndexEntry::Occupied(mut o) => {
                        // we merge both bounds.
                        for bound in bounds {
                            if !o.get().0.contains(&bound) {
                                o.get_mut().0.push(bound);
                            }
                        }
                        for bound_param in bound_params {
                            if !o.get().1.contains(&bound_param) {
                                o.get_mut().1.push(bound_param);
                            }
                        }
                    }
                }
            }
            WherePredicate::RegionPredicate { lifetime, bounds } => {
                match region_predicates.entry(lifetime) {
                    IndexEntry::Vacant(v) => {
                        v.insert(bounds);
                    }
                    IndexEntry::Occupied(mut o) => {
                        // we merge both bounds.
                        for bound in bounds {
                            if !o.get().contains(&bound) {
                                o.get_mut().push(bound);
                            }
                        }
                    }
                }
            }
            WherePredicate::EqPredicate { lhs, rhs } => {
                eq_predicates.push(WherePredicate::EqPredicate { lhs, rhs });
            }
        }
    }

    let mut params = ThinVec::with_capacity(gens.params.len());
    // In this loop, we gather the generic parameters (`<'a, B: 'a>`) and check if they have
    // bounds in the where predicates. If so, we move their bounds into the where predicates
    // while also preventing duplicates.
    for p in gens.params.iter().filter(|p| !is_impl_trait(p) && !is_elided_lifetime(p)) {
        let mut p = clean_generic_param(cx, Some(gens), p);
        match &mut p.kind {
            GenericParamDefKind::Lifetime { outlives } => {
                if let Some(region_pred) = region_predicates.get_mut(&Lifetime(p.name)) {
                    // We merge bounds in the `where` clause.
                    for outlive in outlives.drain(..) {
                        let outlive = GenericBound::Outlives(outlive);
                        if !region_pred.contains(&outlive) {
                            region_pred.push(outlive);
                        }
                    }
                }
            }
            GenericParamDefKind::Type { bounds, synthetic: false, .. } => {
                if let Some(bound_pred) = bound_predicates.get_mut(&Type::Generic(p.name)) {
                    // We merge bounds in the `where` clause.
                    for bound in bounds.drain(..) {
                        if !bound_pred.0.contains(&bound) {
                            bound_pred.0.push(bound);
                        }
                    }
                }
            }
            GenericParamDefKind::Type { .. } | GenericParamDefKind::Const { .. } => {
                // nothing to do here.
            }
        }
        params.push(p);
    }
    params.extend(impl_trait_params);

    Generics {
        params,
        where_predicates: bound_predicates
            .into_iter()
            .map(|(ty, (bounds, bound_params))| WherePredicate::BoundPredicate {
                ty,
                bounds,
                bound_params,
            })
            .chain(
                region_predicates
                    .into_iter()
                    .map(|(lifetime, bounds)| WherePredicate::RegionPredicate { lifetime, bounds }),
            )
            .chain(eq_predicates)
            .collect(),
    }
}

fn clean_ty_generics<'tcx>(
    cx: &mut DocContext<'tcx>,
    gens: &ty::Generics,
    preds: ty::GenericPredicates<'tcx>,
) -> Generics {
    // Don't populate `cx.impl_trait_bounds` before cleaning where clauses,
    // since `clean_predicate` would consume them.
    let mut impl_trait = BTreeMap::<u32, Vec<GenericBound>>::default();

    let params: ThinVec<_> = gens
        .own_params
        .iter()
        .filter(|param| match param.kind {
            ty::GenericParamDefKind::Lifetime => !param.is_anonymous_lifetime(),
            ty::GenericParamDefKind::Type { synthetic, .. } => {
                if param.name == kw::SelfUpper {
                    debug_assert_eq!(param.index, 0);
                    return false;
                }
                if synthetic {
                    impl_trait.insert(param.index, vec![]);
                    return false;
                }
                true
            }
            ty::GenericParamDefKind::Const { .. } => true,
        })
        .map(|param| clean_generic_param_def(param, ParamDefaults::Yes, cx))
        .collect();

    // param index -> [(trait DefId, associated type name & generics, term)]
    let mut impl_trait_proj =
        FxHashMap::<u32, Vec<(DefId, PathSegment, ty::Binder<'_, ty::Term<'_>>)>>::default();

    let where_predicates = preds
        .predicates
        .iter()
        .flat_map(|(pred, _)| {
            let mut projection = None;
            let param_idx = {
                let bound_p = pred.kind();
                match bound_p.skip_binder() {
                    ty::ClauseKind::Trait(pred) if let ty::Param(param) = pred.self_ty().kind() => {
                        Some(param.index)
                    }
                    ty::ClauseKind::TypeOutlives(ty::OutlivesPredicate(ty, _reg))
                        if let ty::Param(param) = ty.kind() =>
                    {
                        Some(param.index)
                    }
                    ty::ClauseKind::Projection(p)
                        if let ty::Param(param) = p.projection_term.self_ty().kind() =>
                    {
                        projection = Some(bound_p.rebind(p));
                        Some(param.index)
                    }
                    _ => None,
                }
            };

            if let Some(param_idx) = param_idx
                && let Some(bounds) = impl_trait.get_mut(&param_idx)
            {
                let pred = clean_predicate(*pred, cx)?;

                bounds.extend(pred.get_bounds().into_iter().flatten().cloned());

                if let Some(proj) = projection
                    && let lhs = clean_projection(
                        proj.map_bound(|p| {
                            // FIXME: This needs to be made resilient for `AliasTerm`s that
                            // are associated consts.
                            p.projection_term.expect_ty(cx.tcx)
                        }),
                        cx,
                        None,
                    )
                    && let Some((_, trait_did, name)) = lhs.projection()
                {
                    impl_trait_proj.entry(param_idx).or_default().push((
                        trait_did,
                        name,
                        proj.map_bound(|p| p.term),
                    ));
                }

                return None;
            }

            Some(pred)
        })
        .collect::<Vec<_>>();

    for (idx, mut bounds) in impl_trait {
        let mut has_sized = false;
        bounds.retain(|b| {
            if b.is_sized_bound(cx) {
                has_sized = true;
                false
            } else {
                true
            }
        });
        if !has_sized {
            bounds.push(GenericBound::maybe_sized(cx));
        }

        // Move trait bounds to the front.
        bounds.sort_by_key(|b| !b.is_trait_bound());

        // Add back a `Sized` bound if there are no *trait* bounds remaining (incl. `?Sized`).
        // Since all potential trait bounds are at the front we can just check the first bound.
        if bounds.first().is_none_or(|b| !b.is_trait_bound()) {
            bounds.insert(0, GenericBound::sized(cx));
        }

        if let Some(proj) = impl_trait_proj.remove(&idx) {
            for (trait_did, name, rhs) in proj {
                let rhs = clean_middle_term(rhs, cx);
                simplify::merge_bounds(cx, &mut bounds, trait_did, name, &rhs);
            }
        }

        cx.impl_trait_bounds.insert(idx.into(), bounds);
    }

    // Now that `cx.impl_trait_bounds` is populated, we can process
    // remaining predicates which could contain `impl Trait`.
    let where_predicates =
        where_predicates.into_iter().flat_map(|p| clean_predicate(*p, cx)).collect();

    let mut generics = Generics { params, where_predicates };
    simplify::sized_bounds(cx, &mut generics);
    generics.where_predicates = simplify::where_clauses(cx, generics.where_predicates);
    generics
}

fn clean_ty_alias_inner_type<'tcx>(
    ty: Ty<'tcx>,
    cx: &mut DocContext<'tcx>,
    ret: &mut Vec<Item>,
) -> Option<TypeAliasInnerType> {
    let ty::Adt(adt_def, args) = ty.kind() else {
        return None;
    };

    if !adt_def.did().is_local() {
        cx.with_param_env(adt_def.did(), |cx| {
            inline::build_impls(cx, adt_def.did(), None, ret);
        });
    }

    Some(if adt_def.is_enum() {
        let variants: rustc_index::IndexVec<_, _> = adt_def
            .variants()
            .iter()
            .map(|variant| clean_variant_def_with_args(variant, args, cx))
            .collect();

        if !adt_def.did().is_local() {
            inline::record_extern_fqn(cx, adt_def.did(), ItemType::Enum);
        }

        TypeAliasInnerType::Enum {
            variants,
            is_non_exhaustive: adt_def.is_variant_list_non_exhaustive(),
        }
    } else {
        let variant = adt_def
            .variants()
            .iter()
            .next()
            .unwrap_or_else(|| bug!("a struct or union should always have one variant def"));

        let fields: Vec<_> =
            clean_variant_def_with_args(variant, args, cx).kind.inner_items().cloned().collect();

        if adt_def.is_struct() {
            if !adt_def.did().is_local() {
                inline::record_extern_fqn(cx, adt_def.did(), ItemType::Struct);
            }
            TypeAliasInnerType::Struct { ctor_kind: variant.ctor_kind(), fields }
        } else {
            if !adt_def.did().is_local() {
                inline::record_extern_fqn(cx, adt_def.did(), ItemType::Union);
            }
            TypeAliasInnerType::Union { fields }
        }
    })
}

fn clean_proc_macro<'tcx>(
    item: &hir::Item<'tcx>,
    name: &mut Symbol,
    kind: MacroKind,
    cx: &mut DocContext<'tcx>,
) -> ItemKind {
    let attrs = cx.tcx.hir_attrs(item.hir_id());
    if kind == MacroKind::Derive
        && let Some(derive_name) =
            hir_attr_lists(attrs, sym::proc_macro_derive).find_map(|mi| mi.ident())
    {
        *name = derive_name.name;
    }

    let mut helpers = Vec::new();
    for mi in hir_attr_lists(attrs, sym::proc_macro_derive) {
        if !mi.has_name(sym::attributes) {
            continue;
        }

        if let Some(list) = mi.meta_item_list() {
            for inner_mi in list {
                if let Some(ident) = inner_mi.ident() {
                    helpers.push(ident.name);
                }
            }
        }
    }
    ProcMacroItem(ProcMacro { kind, helpers })
}

fn clean_fn_or_proc_macro<'tcx>(
    item: &hir::Item<'tcx>,
    sig: &hir::FnSig<'tcx>,
    generics: &hir::Generics<'tcx>,
    body_id: hir::BodyId,
    name: &mut Symbol,
    cx: &mut DocContext<'tcx>,
) -> ItemKind {
    let attrs = cx.tcx.hir_attrs(item.hir_id());
    let macro_kind = attrs.iter().find_map(|a| {
        if a.has_name(sym::proc_macro) {
            Some(MacroKind::Bang)
        } else if a.has_name(sym::proc_macro_derive) {
            Some(MacroKind::Derive)
        } else if a.has_name(sym::proc_macro_attribute) {
            Some(MacroKind::Attr)
        } else {
            None
        }
    });
    match macro_kind {
        Some(kind) => clean_proc_macro(item, name, kind, cx),
        None => {
            let mut func = clean_function(cx, sig, generics, FunctionArgs::Body(body_id));
            clean_fn_decl_legacy_const_generics(&mut func, attrs);
            FunctionItem(func)
        }
    }
}

/// This is needed to make it more "readable" when documenting functions using
/// `rustc_legacy_const_generics`. More information in
/// <https://github.com/rust-lang/rust/issues/83167>.
fn clean_fn_decl_legacy_const_generics(func: &mut Function, attrs: &[hir::Attribute]) {
    for meta_item_list in attrs
        .iter()
        .filter(|a| a.has_name(sym::rustc_legacy_const_generics))
        .filter_map(|a| a.meta_item_list())
    {
        for (pos, literal) in meta_item_list.iter().filter_map(|meta| meta.lit()).enumerate() {
            match literal.kind {
                ast::LitKind::Int(a, _) => {
                    let param = func.generics.params.remove(0);
                    if let GenericParamDef {
                        name,
                        kind: GenericParamDefKind::Const { ty, .. },
                        ..
                    } = param
                    {
                        func.decl
                            .inputs
                            .values
                            .insert(a.get() as _, Argument { name, type_: *ty, is_const: true });
                    } else {
                        panic!("unexpected non const in position {pos}");
                    }
                }
                _ => panic!("invalid arg index"),
            }
        }
    }
}

enum FunctionArgs<'tcx> {
    Body(hir::BodyId),
    Names(&'tcx [Option<Ident>]),
}

fn clean_function<'tcx>(
    cx: &mut DocContext<'tcx>,
    sig: &hir::FnSig<'tcx>,
    generics: &hir::Generics<'tcx>,
    args: FunctionArgs<'tcx>,
) -> Box<Function> {
    let (generics, decl) = enter_impl_trait(cx, |cx| {
        // NOTE: generics must be cleaned before args
        let generics = clean_generics(generics, cx);
        let args = match args {
            FunctionArgs::Body(body_id) => {
                clean_args_from_types_and_body_id(cx, sig.decl.inputs, body_id)
            }
            FunctionArgs::Names(names) => {
                clean_args_from_types_and_names(cx, sig.decl.inputs, names)
            }
        };
        let decl = clean_fn_decl_with_args(cx, sig.decl, Some(&sig.header), args);
        (generics, decl)
    });
    Box::new(Function { decl, generics })
}

fn clean_args_from_types_and_names<'tcx>(
    cx: &mut DocContext<'tcx>,
    types: &[hir::Ty<'tcx>],
    names: &[Option<Ident>],
) -> Arguments {
    fn nonempty_name(ident: &Option<Ident>) -> Option<Symbol> {
        if let Some(ident) = ident
            && ident.name != kw::Underscore
        {
            Some(ident.name)
        } else {
            None
        }
    }

    // If at least one argument has a name, use `_` as the name of unnamed
    // arguments. Otherwise omit argument names.
    let default_name = if names.iter().any(|ident| nonempty_name(ident).is_some()) {
        kw::Underscore
    } else {
        kw::Empty
    };

    Arguments {
        values: types
            .iter()
            .enumerate()
            .map(|(i, ty)| Argument {
                type_: clean_ty(ty, cx),
                name: names.get(i).and_then(nonempty_name).unwrap_or(default_name),
                is_const: false,
            })
            .collect(),
    }
}

fn clean_args_from_types_and_body_id<'tcx>(
    cx: &mut DocContext<'tcx>,
    types: &[hir::Ty<'tcx>],
    body_id: hir::BodyId,
) -> Arguments {
    let body = cx.tcx.hir_body(body_id);

    Arguments {
        values: types
            .iter()
            .zip(body.params)
            .map(|(ty, param)| Argument {
                name: name_from_pat(param.pat),
                type_: clean_ty(ty, cx),
                is_const: false,
            })
            .collect(),
    }
}

fn clean_fn_decl_with_args<'tcx>(
    cx: &mut DocContext<'tcx>,
    decl: &hir::FnDecl<'tcx>,
    header: Option<&hir::FnHeader>,
    args: Arguments,
) -> FnDecl {
    let mut output = match decl.output {
        hir::FnRetTy::Return(typ) => clean_ty(typ, cx),
        hir::FnRetTy::DefaultReturn(..) => Type::Tuple(Vec::new()),
    };
    if let Some(header) = header
        && header.is_async()
    {
        output = output.sugared_async_return_type();
    }
    FnDecl { inputs: args, output, c_variadic: decl.c_variadic }
}

fn clean_poly_fn_sig<'tcx>(
    cx: &mut DocContext<'tcx>,
    did: Option<DefId>,
    sig: ty::PolyFnSig<'tcx>,
) -> FnDecl {
    let mut names = did.map_or(&[] as &[_], |did| cx.tcx.fn_arg_names(did)).iter();

    // We assume all empty tuples are default return type. This theoretically can discard `-> ()`,
    // but shouldn't change any code meaning.
    let mut output = clean_middle_ty(sig.output(), cx, None, None);

    // If the return type isn't an `impl Trait`, we can safely assume that this
    // function isn't async without needing to execute the query `asyncness` at
    // all which gives us a noticeable performance boost.
    if let Some(did) = did
        && let Type::ImplTrait(_) = output
        && cx.tcx.asyncness(did).is_async()
    {
        output = output.sugared_async_return_type();
    }

    FnDecl {
        output,
        c_variadic: sig.skip_binder().c_variadic,
        inputs: Arguments {
            values: sig
                .inputs()
                .iter()
                .map(|t| Argument {
                    type_: clean_middle_ty(t.map_bound(|t| *t), cx, None, None),
                    name: if let Some(Some(ident)) = names.next() {
                        ident.name
                    } else {
                        kw::Underscore
                    },
                    is_const: false,
                })
                .collect(),
        },
    }
}

fn clean_trait_ref<'tcx>(trait_ref: &hir::TraitRef<'tcx>, cx: &mut DocContext<'tcx>) -> Path {
    let path = clean_path(trait_ref.path, cx);
    register_res(cx, path.res);
    path
}

fn clean_poly_trait_ref<'tcx>(
    poly_trait_ref: &hir::PolyTraitRef<'tcx>,
    cx: &mut DocContext<'tcx>,
) -> PolyTrait {
    PolyTrait {
        trait_: clean_trait_ref(&poly_trait_ref.trait_ref, cx),
        generic_params: poly_trait_ref
            .bound_generic_params
            .iter()
            .filter(|p| !is_elided_lifetime(p))
            .map(|x| clean_generic_param(cx, None, x))
            .collect(),
    }
}

fn clean_trait_item<'tcx>(trait_item: &hir::TraitItem<'tcx>, cx: &mut DocContext<'tcx>) -> Item {
    let local_did = trait_item.owner_id.to_def_id();
    cx.with_param_env(local_did, |cx| {
        let inner = match trait_item.kind {
            hir::TraitItemKind::Const(ty, Some(default)) => {
                ProvidedAssocConstItem(Box::new(Constant {
                    generics: enter_impl_trait(cx, |cx| clean_generics(trait_item.generics, cx)),
                    kind: ConstantKind::Local { def_id: local_did, body: default },
                    type_: clean_ty(ty, cx),
                }))
            }
            hir::TraitItemKind::Const(ty, None) => {
                let generics = enter_impl_trait(cx, |cx| clean_generics(trait_item.generics, cx));
                RequiredAssocConstItem(generics, Box::new(clean_ty(ty, cx)))
            }
            hir::TraitItemKind::Fn(ref sig, hir::TraitFn::Provided(body)) => {
                let m = clean_function(cx, sig, trait_item.generics, FunctionArgs::Body(body));
                MethodItem(m, None)
            }
            hir::TraitItemKind::Fn(ref sig, hir::TraitFn::Required(names)) => {
                let m = clean_function(cx, sig, trait_item.generics, FunctionArgs::Names(names));
                RequiredMethodItem(m)
            }
            hir::TraitItemKind::Type(bounds, Some(default)) => {
                let generics = enter_impl_trait(cx, |cx| clean_generics(trait_item.generics, cx));
                let bounds = bounds.iter().filter_map(|x| clean_generic_bound(x, cx)).collect();
                let item_type =
                    clean_middle_ty(ty::Binder::dummy(lower_ty(cx.tcx, default)), cx, None, None);
                AssocTypeItem(
                    Box::new(TypeAlias {
                        type_: clean_ty(default, cx),
                        generics,
                        inner_type: None,
                        item_type: Some(item_type),
                    }),
                    bounds,
                )
            }
            hir::TraitItemKind::Type(bounds, None) => {
                let generics = enter_impl_trait(cx, |cx| clean_generics(trait_item.generics, cx));
                let bounds = bounds.iter().filter_map(|x| clean_generic_bound(x, cx)).collect();
                RequiredAssocTypeItem(generics, bounds)
            }
        };
        Item::from_def_id_and_parts(local_did, Some(trait_item.ident.name), inner, cx)
    })
}

pub(crate) fn clean_impl_item<'tcx>(
    impl_: &hir::ImplItem<'tcx>,
    cx: &mut DocContext<'tcx>,
) -> Item {
    let local_did = impl_.owner_id.to_def_id();
    cx.with_param_env(local_did, |cx| {
        let inner = match impl_.kind {
            hir::ImplItemKind::Const(ty, expr) => ImplAssocConstItem(Box::new(Constant {
                generics: clean_generics(impl_.generics, cx),
                kind: ConstantKind::Local { def_id: local_did, body: expr },
                type_: clean_ty(ty, cx),
            })),
            hir::ImplItemKind::Fn(ref sig, body) => {
                let m = clean_function(cx, sig, impl_.generics, FunctionArgs::Body(body));
                let defaultness = cx.tcx.defaultness(impl_.owner_id);
                MethodItem(m, Some(defaultness))
            }
            hir::ImplItemKind::Type(hir_ty) => {
                let type_ = clean_ty(hir_ty, cx);
                let generics = clean_generics(impl_.generics, cx);
                let item_type =
                    clean_middle_ty(ty::Binder::dummy(lower_ty(cx.tcx, hir_ty)), cx, None, None);
                AssocTypeItem(
                    Box::new(TypeAlias {
                        type_,
                        generics,
                        inner_type: None,
                        item_type: Some(item_type),
                    }),
                    Vec::new(),
                )
            }
        };

        Item::from_def_id_and_parts(local_did, Some(impl_.ident.name), inner, cx)
    })
}

pub(crate) fn clean_middle_assoc_item(assoc_item: &ty::AssocItem, cx: &mut DocContext<'_>) -> Item {
    let tcx = cx.tcx;
    let kind = match assoc_item.kind {
        ty::AssocKind::Const => {
            let ty = clean_middle_ty(
                ty::Binder::dummy(tcx.type_of(assoc_item.def_id).instantiate_identity()),
                cx,
                Some(assoc_item.def_id),
                None,
            );

            let mut generics = clean_ty_generics(
                cx,
                tcx.generics_of(assoc_item.def_id),
                tcx.explicit_predicates_of(assoc_item.def_id),
            );
            simplify::move_bounds_to_generic_parameters(&mut generics);

            match assoc_item.container {
                ty::AssocItemContainer::Impl => ImplAssocConstItem(Box::new(Constant {
                    generics,
                    kind: ConstantKind::Extern { def_id: assoc_item.def_id },
                    type_: ty,
                })),
                ty::AssocItemContainer::Trait => {
                    if tcx.defaultness(assoc_item.def_id).has_value() {
                        ProvidedAssocConstItem(Box::new(Constant {
                            generics,
                            kind: ConstantKind::Extern { def_id: assoc_item.def_id },
                            type_: ty,
                        }))
                    } else {
                        RequiredAssocConstItem(generics, Box::new(ty))
                    }
                }
            }
        }
        ty::AssocKind::Fn => {
            let mut item = inline::build_function(cx, assoc_item.def_id);

            if assoc_item.fn_has_self_parameter {
                let self_ty = match assoc_item.container {
                    ty::AssocItemContainer::Impl => {
                        tcx.type_of(assoc_item.container_id(tcx)).instantiate_identity()
                    }
                    ty::AssocItemContainer::Trait => tcx.types.self_param,
                };
                let self_arg_ty =
                    tcx.fn_sig(assoc_item.def_id).instantiate_identity().input(0).skip_binder();
                if self_arg_ty == self_ty {
                    item.decl.inputs.values[0].type_ = SelfTy;
                } else if let ty::Ref(_, ty, _) = *self_arg_ty.kind()
                    && ty == self_ty
                {
                    match item.decl.inputs.values[0].type_ {
                        BorrowedRef { ref mut type_, .. } => **type_ = SelfTy,
                        _ => unreachable!(),
                    }
                }
            }

            let provided = match assoc_item.container {
                ty::AssocItemContainer::Impl => true,
                ty::AssocItemContainer::Trait => assoc_item.defaultness(tcx).has_value(),
            };
            if provided {
                let defaultness = match assoc_item.container {
                    ty::AssocItemContainer::Impl => Some(assoc_item.defaultness(tcx)),
                    ty::AssocItemContainer::Trait => None,
                };
                MethodItem(item, defaultness)
            } else {
                RequiredMethodItem(item)
            }
        }
        ty::AssocKind::Type => {
            let my_name = assoc_item.name;

            fn param_eq_arg(param: &GenericParamDef, arg: &GenericArg) -> bool {
                match (&param.kind, arg) {
                    (GenericParamDefKind::Type { .. }, GenericArg::Type(Type::Generic(ty)))
                        if *ty == param.name =>
                    {
                        true
                    }
                    (GenericParamDefKind::Lifetime { .. }, GenericArg::Lifetime(Lifetime(lt)))
                        if *lt == param.name =>
                    {
                        true
                    }
                    (GenericParamDefKind::Const { .. }, GenericArg::Const(c)) => match &**c {
                        ConstantKind::TyConst { expr } => **expr == *param.name.as_str(),
                        _ => false,
                    },
                    _ => false,
                }
            }

            let mut predicates = tcx.explicit_predicates_of(assoc_item.def_id).predicates;
            if let ty::AssocItemContainer::Trait = assoc_item.container {
                let bounds = tcx.explicit_item_bounds(assoc_item.def_id).iter_identity_copied();
                predicates = tcx.arena.alloc_from_iter(bounds.chain(predicates.iter().copied()));
            }
            let mut generics = clean_ty_generics(
                cx,
                tcx.generics_of(assoc_item.def_id),
                ty::GenericPredicates { parent: None, predicates },
            );
            simplify::move_bounds_to_generic_parameters(&mut generics);

            if let ty::AssocItemContainer::Trait = assoc_item.container {
                // Move bounds that are (likely) directly attached to the associated type
                // from the where-clause to the associated type.
                // There is no guarantee that this is what the user actually wrote but we have
                // no way of knowing.
                let mut bounds: Vec<GenericBound> = Vec::new();
                generics.where_predicates.retain_mut(|pred| match *pred {
                    WherePredicate::BoundPredicate {
                        ty:
                            QPath(box QPathData {
                                ref assoc,
                                ref self_type,
                                trait_: Some(ref trait_),
                                ..
                            }),
                        bounds: ref mut pred_bounds,
                        ..
                    } => {
                        if assoc.name != my_name {
                            return true;
                        }
                        if trait_.def_id() != assoc_item.container_id(tcx) {
                            return true;
                        }
                        if *self_type != SelfTy {
                            return true;
                        }
                        match &assoc.args {
                            GenericArgs::AngleBracketed { args, constraints } => {
                                if !constraints.is_empty()
                                    || generics
                                        .params
                                        .iter()
                                        .zip(args.iter())
                                        .any(|(param, arg)| !param_eq_arg(param, arg))
                                {
                                    return true;
                                }
                            }
                            GenericArgs::Parenthesized { .. } => {
                                // The only time this happens is if we're inside the rustdoc for Fn(),
                                // which only has one associated type, which is not a GAT, so whatever.
                            }
                            GenericArgs::ReturnTypeNotation => {
                                // Never move these.
                            }
                        }
                        bounds.extend(mem::take(pred_bounds));
                        false
                    }
                    _ => true,
                });
                // Our Sized/?Sized bound didn't get handled when creating the generics
                // because we didn't actually get our whole set of bounds until just now
                // (some of them may have come from the trait). If we do have a sized
                // bound, we remove it, and if we don't then we add the `?Sized` bound
                // at the end.
                match bounds.iter().position(|b| b.is_sized_bound(cx)) {
                    Some(i) => {
                        bounds.remove(i);
                    }
                    None => bounds.push(GenericBound::maybe_sized(cx)),
                }

                if tcx.defaultness(assoc_item.def_id).has_value() {
                    AssocTypeItem(
                        Box::new(TypeAlias {
                            type_: clean_middle_ty(
                                ty::Binder::dummy(
                                    tcx.type_of(assoc_item.def_id).instantiate_identity(),
                                ),
                                cx,
                                Some(assoc_item.def_id),
                                None,
                            ),
                            generics,
                            inner_type: None,
                            item_type: None,
                        }),
                        bounds,
                    )
                } else {
                    RequiredAssocTypeItem(generics, bounds)
                }
            } else {
                AssocTypeItem(
                    Box::new(TypeAlias {
                        type_: clean_middle_ty(
                            ty::Binder::dummy(
                                tcx.type_of(assoc_item.def_id).instantiate_identity(),
                            ),
                            cx,
                            Some(assoc_item.def_id),
                            None,
                        ),
                        generics,
                        inner_type: None,
                        item_type: None,
                    }),
                    // Associated types inside trait or inherent impls are not allowed to have
                    // item bounds. Thus we don't attempt to move any bounds there.
                    Vec::new(),
                )
            }
        }
    };

    Item::from_def_id_and_parts(assoc_item.def_id, Some(assoc_item.name), kind, cx)
}

fn first_non_private_clean_path<'tcx>(
    cx: &mut DocContext<'tcx>,
    path: &hir::Path<'tcx>,
    new_path_segments: &'tcx [hir::PathSegment<'tcx>],
    new_path_span: rustc_span::Span,
) -> Path {
    let new_hir_path =
        hir::Path { segments: new_path_segments, res: path.res, span: new_path_span };
    let mut new_clean_path = clean_path(&new_hir_path, cx);
    // In here we need to play with the path data one last time to provide it the
    // missing `args` and `res` of the final `Path` we get, which, since it comes
    // from a re-export, doesn't have the generics that were originally there, so
    // we add them by hand.
    if let Some(path_last) = path.segments.last().as_ref()
        && let Some(new_path_last) = new_clean_path.segments[..].last_mut()
        && let Some(path_last_args) = path_last.args.as_ref()
        && path_last.args.is_some()
    {
        assert!(new_path_last.args.is_empty());
        new_path_last.args = clean_generic_args(path_last_args, cx);
    }
    new_clean_path
}

/// The goal of this function is to return the first `Path` which is not private (ie not private
/// or `doc(hidden)`). If it's not possible, it'll return the "end type".
///
/// If the path is not a re-export or is public, it'll return `None`.
fn first_non_private<'tcx>(
    cx: &mut DocContext<'tcx>,
    hir_id: hir::HirId,
    path: &hir::Path<'tcx>,
) -> Option<Path> {
    let target_def_id = path.res.opt_def_id()?;
    let (parent_def_id, ident) = match &path.segments {
        [] => return None,
        // Relative paths are available in the same scope as the owner.
        [leaf] => (cx.tcx.local_parent(hir_id.owner.def_id), leaf.ident),
        // So are self paths.
        [parent, leaf] if parent.ident.name == kw::SelfLower => {
            (cx.tcx.local_parent(hir_id.owner.def_id), leaf.ident)
        }
        // Crate paths are not. We start from the crate root.
        [parent, leaf] if matches!(parent.ident.name, kw::Crate | kw::PathRoot) => {
            (LOCAL_CRATE.as_def_id().as_local()?, leaf.ident)
        }
        [parent, leaf] if parent.ident.name == kw::Super => {
            let parent_mod = cx.tcx.parent_module(hir_id);
            if let Some(super_parent) = cx.tcx.opt_local_parent(parent_mod.to_local_def_id()) {
                (super_parent, leaf.ident)
            } else {
                // If we can't find the parent of the parent, then the parent is already the crate.
                (LOCAL_CRATE.as_def_id().as_local()?, leaf.ident)
            }
        }
        // Absolute paths are not. We start from the parent of the item.
        [.., parent, leaf] => (parent.res.opt_def_id()?.as_local()?, leaf.ident),
    };
    // First we try to get the `DefId` of the item.
    for child in
        cx.tcx.module_children_local(parent_def_id).iter().filter(move |c| c.ident == ident)
    {
        if let Res::Def(DefKind::Ctor(..), _) | Res::SelfCtor(..) = child.res {
            continue;
        }

        if let Some(def_id) = child.res.opt_def_id()
            && target_def_id == def_id
        {
            let mut last_path_res = None;
            'reexps: for reexp in child.reexport_chain.iter() {
                if let Some(use_def_id) = reexp.id()
                    && let Some(local_use_def_id) = use_def_id.as_local()
                    && let hir::Node::Item(item) = cx.tcx.hir_node_by_def_id(local_use_def_id)
                    && let hir::ItemKind::Use(path, hir::UseKind::Single(_)) = item.kind
                {
                    for res in &path.res {
                        if let Res::Def(DefKind::Ctor(..), _) | Res::SelfCtor(..) = res {
                            continue;
                        }
                        if (cx.render_options.document_hidden ||
                            !cx.tcx.is_doc_hidden(use_def_id)) &&
                            // We never check for "cx.render_options.document_private"
                            // because if a re-export is not fully public, it's never
                            // documented.
                            cx.tcx.local_visibility(local_use_def_id).is_public()
                        {
                            break 'reexps;
                        }
                        last_path_res = Some((path, res));
                        continue 'reexps;
                    }
                }
            }
            if !child.reexport_chain.is_empty() {
                // So in here, we use the data we gathered from iterating the reexports. If
                // `last_path_res` is set, it can mean two things:
                //
                // 1. We found a public reexport.
                // 2. We didn't find a public reexport so it's the "end type" path.
                if let Some((new_path, _)) = last_path_res {
                    return Some(first_non_private_clean_path(
                        cx,
                        path,
                        new_path.segments,
                        new_path.span,
                    ));
                }
                // If `last_path_res` is `None`, it can mean two things:
                //
                // 1. The re-export is public, no need to change anything, just use the path as is.
                // 2. Nothing was found, so let's just return the original path.
                return None;
            }
        }
    }
    None
}

fn clean_qpath<'tcx>(hir_ty: &hir::Ty<'tcx>, cx: &mut DocContext<'tcx>) -> Type {
    let hir::Ty { hir_id, span, ref kind } = *hir_ty;
    let hir::TyKind::Path(qpath) = kind else { unreachable!() };

    match qpath {
        hir::QPath::Resolved(None, path) => {
            if let Res::Def(DefKind::TyParam, did) = path.res {
                if let Some(new_ty) = cx.args.get(&did).and_then(|p| p.as_ty()).cloned() {
                    return new_ty;
                }
                if let Some(bounds) = cx.impl_trait_bounds.remove(&did.into()) {
                    return ImplTrait(bounds);
                }
            }

            if let Some(expanded) = maybe_expand_private_type_alias(cx, path) {
                expanded
            } else {
                // First we check if it's a private re-export.
                let path = if let Some(path) = first_non_private(cx, hir_id, path) {
                    path
                } else {
                    clean_path(path, cx)
                };
                resolve_type(cx, path)
            }
        }
        hir::QPath::Resolved(Some(qself), p) => {
            // Try to normalize `<X as Y>::T` to a type
            let ty = lower_ty(cx.tcx, hir_ty);
            // `hir_to_ty` can return projection types with escaping vars for GATs, e.g. `<() as Trait>::Gat<'_>`
            if !ty.has_escaping_bound_vars()
                && let Some(normalized_value) = normalize(cx, ty::Binder::dummy(ty))
            {
                return clean_middle_ty(normalized_value, cx, None, None);
            }

            let trait_segments = &p.segments[..p.segments.len() - 1];
            let trait_def = cx.tcx.associated_item(p.res.def_id()).container_id(cx.tcx);
            let trait_ = self::Path {
                res: Res::Def(DefKind::Trait, trait_def),
                segments: trait_segments.iter().map(|x| clean_path_segment(x, cx)).collect(),
            };
            register_res(cx, trait_.res);
            let self_def_id = DefId::local(qself.hir_id.owner.def_id.local_def_index);
            let self_type = clean_ty(qself, cx);
            let should_show_cast = compute_should_show_cast(Some(self_def_id), &trait_, &self_type);
            Type::QPath(Box::new(QPathData {
                assoc: clean_path_segment(p.segments.last().expect("segments were empty"), cx),
                should_show_cast,
                self_type,
                trait_: Some(trait_),
            }))
        }
        hir::QPath::TypeRelative(qself, segment) => {
            let ty = lower_ty(cx.tcx, hir_ty);
            let self_type = clean_ty(qself, cx);

            let (trait_, should_show_cast) = match ty.kind() {
                ty::Alias(ty::Projection, proj) => {
                    let res = Res::Def(DefKind::Trait, proj.trait_ref(cx.tcx).def_id);
                    let trait_ = clean_path(&hir::Path { span, res, segments: &[] }, cx);
                    register_res(cx, trait_.res);
                    let self_def_id = res.opt_def_id();
                    let should_show_cast =
                        compute_should_show_cast(self_def_id, &trait_, &self_type);

                    (Some(trait_), should_show_cast)
                }
                ty::Alias(ty::Inherent, _) => (None, false),
                // Rustdoc handles `ty::Error`s by turning them into `Type::Infer`s.
                ty::Error(_) => return Type::Infer,
                _ => bug!("clean: expected associated type, found `{ty:?}`"),
            };

            Type::QPath(Box::new(QPathData {
                assoc: clean_path_segment(segment, cx),
                should_show_cast,
                self_type,
                trait_,
            }))
        }
        hir::QPath::LangItem(..) => bug!("clean: requiring documentation of lang item"),
    }
}

fn maybe_expand_private_type_alias<'tcx>(
    cx: &mut DocContext<'tcx>,
    path: &hir::Path<'tcx>,
) -> Option<Type> {
    let Res::Def(DefKind::TyAlias, def_id) = path.res else { return None };
    // Substitute private type aliases
    let def_id = def_id.as_local()?;
    let alias = if !cx.cache.effective_visibilities.is_exported(cx.tcx, def_id.to_def_id())
        && !cx.current_type_aliases.contains_key(&def_id.to_def_id())
    {
        &cx.tcx.hir_expect_item(def_id).kind
    } else {
        return None;
    };
    let hir::ItemKind::TyAlias(_, ty, generics) = alias else { return None };

    let final_seg = &path.segments.last().expect("segments were empty");
    let mut args = DefIdMap::default();
    let generic_args = final_seg.args();

    let mut indices: hir::GenericParamCount = Default::default();
    for param in generics.params.iter() {
        match param.kind {
            hir::GenericParamKind::Lifetime { .. } => {
                let mut j = 0;
                let lifetime = generic_args.args.iter().find_map(|arg| match arg {
                    hir::GenericArg::Lifetime(lt) => {
                        if indices.lifetimes == j {
                            return Some(lt);
                        }
                        j += 1;
                        None
                    }
                    _ => None,
                });
                if let Some(lt) = lifetime {
                    let lt = if !lt.is_anonymous() {
                        clean_lifetime(lt, cx)
                    } else {
                        Lifetime::elided()
                    };
                    args.insert(param.def_id.to_def_id(), GenericArg::Lifetime(lt));
                }
                indices.lifetimes += 1;
            }
            hir::GenericParamKind::Type { ref default, .. } => {
                let mut j = 0;
                let type_ = generic_args.args.iter().find_map(|arg| match arg {
                    hir::GenericArg::Type(ty) => {
                        if indices.types == j {
                            return Some(ty.as_unambig_ty());
                        }
                        j += 1;
                        None
                    }
                    _ => None,
                });
                if let Some(ty) = type_.or(*default) {
                    args.insert(param.def_id.to_def_id(), GenericArg::Type(clean_ty(ty, cx)));
                }
                indices.types += 1;
            }
            // FIXME(#82852): Instantiate const parameters.
            hir::GenericParamKind::Const { .. } => {}
        }
    }

    Some(cx.enter_alias(args, def_id.to_def_id(), |cx| {
        cx.with_param_env(def_id.to_def_id(), |cx| clean_ty(ty, cx))
    }))
}

pub(crate) fn clean_ty<'tcx>(ty: &hir::Ty<'tcx>, cx: &mut DocContext<'tcx>) -> Type {
    use rustc_hir::*;

    match ty.kind {
        TyKind::Never => Primitive(PrimitiveType::Never),
        TyKind::Ptr(ref m) => RawPointer(m.mutbl, Box::new(clean_ty(m.ty, cx))),
        TyKind::Ref(l, ref m) => {
            let lifetime = if l.is_anonymous() { None } else { Some(clean_lifetime(l, cx)) };
            BorrowedRef { lifetime, mutability: m.mutbl, type_: Box::new(clean_ty(m.ty, cx)) }
        }
        TyKind::Slice(ty) => Slice(Box::new(clean_ty(ty, cx))),
        TyKind::Pat(ty, pat) => Type::Pat(Box::new(clean_ty(ty, cx)), format!("{pat:?}").into()),
        TyKind::Array(ty, const_arg) => {
            // NOTE(min_const_generics): We can't use `const_eval_poly` for constants
            // as we currently do not supply the parent generics to anonymous constants
            // but do allow `ConstKind::Param`.
            //
            // `const_eval_poly` tries to first substitute generic parameters which
            // results in an ICE while manually constructing the constant and using `eval`
            // does nothing for `ConstKind::Param`.
            let length = match const_arg.kind {
                hir::ConstArgKind::Infer(..) => "_".to_string(),
                hir::ConstArgKind::Anon(hir::AnonConst { def_id, .. }) => {
                    let ct = lower_const_arg_for_rustdoc(cx.tcx, const_arg, FeedConstTy::No);
                    let typing_env = ty::TypingEnv::post_analysis(cx.tcx, *def_id);
                    let ct = cx.tcx.normalize_erasing_regions(typing_env, ct);
                    print_const(cx, ct)
                }
                hir::ConstArgKind::Path(..) => {
                    let ct = lower_const_arg_for_rustdoc(cx.tcx, const_arg, FeedConstTy::No);
                    print_const(cx, ct)
                }
            };
            Array(Box::new(clean_ty(ty, cx)), length.into())
        }
        TyKind::Tup(tys) => Tuple(tys.iter().map(|ty| clean_ty(ty, cx)).collect()),
        TyKind::OpaqueDef(ty) => {
            ImplTrait(ty.bounds.iter().filter_map(|x| clean_generic_bound(x, cx)).collect())
        }
        TyKind::Path(_) => clean_qpath(ty, cx),
        TyKind::TraitObject(bounds, lifetime) => {
            let bounds = bounds.iter().map(|bound| clean_poly_trait_ref(bound, cx)).collect();
            let lifetime = if !lifetime.is_elided() {
                Some(clean_lifetime(lifetime.pointer(), cx))
            } else {
                None
            };
            DynTrait(bounds, lifetime)
        }
        TyKind::BareFn(barefn) => BareFunction(Box::new(clean_bare_fn_ty(barefn, cx))),
        TyKind::UnsafeBinder(unsafe_binder_ty) => {
            UnsafeBinder(Box::new(clean_unsafe_binder_ty(unsafe_binder_ty, cx)))
        }
        // Rustdoc handles `TyKind::Err`s by turning them into `Type::Infer`s.
        TyKind::Infer(())
        | TyKind::Err(_)
        | TyKind::Typeof(..)
        | TyKind::InferDelegation(..)
        | TyKind::TraitAscription(_) => Infer,
    }
}

/// Returns `None` if the type could not be normalized
fn normalize<'tcx>(
    cx: &DocContext<'tcx>,
    ty: ty::Binder<'tcx, Ty<'tcx>>,
) -> Option<ty::Binder<'tcx, Ty<'tcx>>> {
    // HACK: low-churn fix for #79459 while we wait for a trait normalization fix
    if !cx.tcx.sess.opts.unstable_opts.normalize_docs {
        return None;
    }

    use rustc_middle::traits::ObligationCause;
    use rustc_trait_selection::infer::TyCtxtInferExt;
    use rustc_trait_selection::traits::query::normalize::QueryNormalizeExt;

    // Try to normalize `<X as Y>::T` to a type
    let infcx = cx.tcx.infer_ctxt().build(TypingMode::non_body_analysis());
    let normalized = infcx
        .at(&ObligationCause::dummy(), cx.param_env)
        .query_normalize(ty)
        .map(|resolved| infcx.resolve_vars_if_possible(resolved.value));
    match normalized {
        Ok(normalized_value) => {
            debug!("normalized {ty:?} to {normalized_value:?}");
            Some(normalized_value)
        }
        Err(err) => {
            debug!("failed to normalize {ty:?}: {err:?}");
            None
        }
    }
}

fn clean_trait_object_lifetime_bound<'tcx>(
    region: ty::Region<'tcx>,
    container: Option<ContainerTy<'_, 'tcx>>,
    preds: &'tcx ty::List<ty::PolyExistentialPredicate<'tcx>>,
    tcx: TyCtxt<'tcx>,
) -> Option<Lifetime> {
    if can_elide_trait_object_lifetime_bound(region, container, preds, tcx) {
        return None;
    }

    // Since there is a semantic difference between an implicitly elided (i.e. "defaulted") object
    // lifetime and an explicitly elided object lifetime (`'_`), we intentionally don't hide the
    // latter contrary to `clean_middle_region`.
    match region.kind() {
        ty::ReStatic => Some(Lifetime::statik()),
        ty::ReEarlyParam(region) => Some(Lifetime(region.name)),
        ty::ReBound(_, ty::BoundRegion { kind: ty::BoundRegionKind::Named(_, name), .. }) => {
            Some(Lifetime(name))
        }
        ty::ReBound(..)
        | ty::ReLateParam(_)
        | ty::ReVar(_)
        | ty::RePlaceholder(_)
        | ty::ReErased
        | ty::ReError(_) => None,
    }
}

fn can_elide_trait_object_lifetime_bound<'tcx>(
    region: ty::Region<'tcx>,
    container: Option<ContainerTy<'_, 'tcx>>,
    preds: &'tcx ty::List<ty::PolyExistentialPredicate<'tcx>>,
    tcx: TyCtxt<'tcx>,
) -> bool {
    // Below we quote extracts from https://doc.rust-lang.org/stable/reference/lifetime-elision.html#default-trait-object-lifetimes

    // > If the trait object is used as a type argument of a generic type then the containing type is
    // > first used to try to infer a bound.
    let default = container
        .map_or(ObjectLifetimeDefault::Empty, |container| container.object_lifetime_default(tcx));

    // > If there is a unique bound from the containing type then that is the default
    // If there is a default object lifetime and the given region is lexically equal to it, elide it.
    match default {
        ObjectLifetimeDefault::Static => return region.kind() == ty::ReStatic,
        // FIXME(fmease): Don't compare lexically but respect de Bruijn indices etc. to handle shadowing correctly.
        ObjectLifetimeDefault::Arg(default) => return region.get_name() == default.get_name(),
        // > If there is more than one bound from the containing type then an explicit bound must be specified
        // Due to ambiguity there is no default trait-object lifetime and thus elision is impossible.
        // Don't elide the lifetime.
        ObjectLifetimeDefault::Ambiguous => return false,
        // There is no meaningful bound. Further processing is needed...
        ObjectLifetimeDefault::Empty => {}
    }

    // > If neither of those rules apply, then the bounds on the trait are used:
    match *object_region_bounds(tcx, preds) {
        // > If the trait has no lifetime bounds, then the lifetime is inferred in expressions
        // > and is 'static outside of expressions.
        // FIXME: If we are in an expression context (i.e. fn bodies and const exprs) then the default is
        // `'_` and not `'static`. Only if we are in a non-expression one, the default is `'static`.
        // Note however that at the time of this writing it should be fine to disregard this subtlety
        // as we neither render const exprs faithfully anyway (hiding them in some places or using `_` instead)
        // nor show the contents of fn bodies.
        [] => region.kind() == ty::ReStatic,
        // > If the trait is defined with a single lifetime bound then that bound is used.
        // > If 'static is used for any lifetime bound then 'static is used.
        // FIXME(fmease): Don't compare lexically but respect de Bruijn indices etc. to handle shadowing correctly.
        [object_region] => object_region.get_name() == region.get_name(),
        // There are several distinct trait regions and none are `'static`.
        // Due to ambiguity there is no default trait-object lifetime and thus elision is impossible.
        // Don't elide the lifetime.
        _ => false,
    }
}

#[derive(Debug)]
pub(crate) enum ContainerTy<'a, 'tcx> {
    Ref(ty::Region<'tcx>),
    Regular {
        ty: DefId,
        /// The arguments *have* to contain an arg for the self type if the corresponding generics
        /// contain a self type.
        args: ty::Binder<'tcx, &'a [ty::GenericArg<'tcx>]>,
        arg: usize,
    },
}

impl<'tcx> ContainerTy<'_, 'tcx> {
    fn object_lifetime_default(self, tcx: TyCtxt<'tcx>) -> ObjectLifetimeDefault<'tcx> {
        match self {
            Self::Ref(region) => ObjectLifetimeDefault::Arg(region),
            Self::Regular { ty: container, args, arg: index } => {
                let (DefKind::Struct
                | DefKind::Union
                | DefKind::Enum
                | DefKind::TyAlias
                | DefKind::Trait) = tcx.def_kind(container)
                else {
                    return ObjectLifetimeDefault::Empty;
                };

                let generics = tcx.generics_of(container);
                debug_assert_eq!(generics.parent_count, 0);

                let param = generics.own_params[index].def_id;
                let default = tcx.object_lifetime_default(param);
                match default {
                    rbv::ObjectLifetimeDefault::Param(lifetime) => {
                        // The index is relative to the parent generics but since we don't have any,
                        // we don't need to translate it.
                        let index = generics.param_def_id_to_index[&lifetime];
                        let arg = args.skip_binder()[index as usize].expect_region();
                        ObjectLifetimeDefault::Arg(arg)
                    }
                    rbv::ObjectLifetimeDefault::Empty => ObjectLifetimeDefault::Empty,
                    rbv::ObjectLifetimeDefault::Static => ObjectLifetimeDefault::Static,
                    rbv::ObjectLifetimeDefault::Ambiguous => ObjectLifetimeDefault::Ambiguous,
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum ObjectLifetimeDefault<'tcx> {
    Empty,
    Static,
    Ambiguous,
    Arg(ty::Region<'tcx>),
}

#[instrument(level = "trace", skip(cx), ret)]
pub(crate) fn clean_middle_ty<'tcx>(
    bound_ty: ty::Binder<'tcx, Ty<'tcx>>,
    cx: &mut DocContext<'tcx>,
    parent_def_id: Option<DefId>,
    container: Option<ContainerTy<'_, 'tcx>>,
) -> Type {
    let bound_ty = normalize(cx, bound_ty).unwrap_or(bound_ty);
    match *bound_ty.skip_binder().kind() {
        ty::Never => Primitive(PrimitiveType::Never),
        ty::Bool => Primitive(PrimitiveType::Bool),
        ty::Char => Primitive(PrimitiveType::Char),
        ty::Int(int_ty) => Primitive(int_ty.into()),
        ty::Uint(uint_ty) => Primitive(uint_ty.into()),
        ty::Float(float_ty) => Primitive(float_ty.into()),
        ty::Str => Primitive(PrimitiveType::Str),
        ty::Slice(ty) => Slice(Box::new(clean_middle_ty(bound_ty.rebind(ty), cx, None, None))),
        ty::Pat(ty, pat) => Type::Pat(
            Box::new(clean_middle_ty(bound_ty.rebind(ty), cx, None, None)),
            format!("{pat:?}").into_boxed_str(),
        ),
        ty::Array(ty, n) => {
            let n = cx.tcx.normalize_erasing_regions(cx.typing_env(), n);
            let n = print_const(cx, n);
            Array(Box::new(clean_middle_ty(bound_ty.rebind(ty), cx, None, None)), n.into())
        }
        ty::RawPtr(ty, mutbl) => {
            RawPointer(mutbl, Box::new(clean_middle_ty(bound_ty.rebind(ty), cx, None, None)))
        }
        ty::Ref(r, ty, mutbl) => BorrowedRef {
            lifetime: clean_middle_region(r),
            mutability: mutbl,
            type_: Box::new(clean_middle_ty(
                bound_ty.rebind(ty),
                cx,
                None,
                Some(ContainerTy::Ref(r)),
            )),
        },
        ty::FnDef(..) | ty::FnPtr(..) => {
            // FIXME: should we merge the outer and inner binders somehow?
            let sig = bound_ty.skip_binder().fn_sig(cx.tcx);
            let decl = clean_poly_fn_sig(cx, None, sig);
            let generic_params = clean_bound_vars(sig.bound_vars());

            BareFunction(Box::new(BareFunctionDecl {
                safety: sig.safety(),
                generic_params,
                decl,
                abi: sig.abi(),
            }))
        }
        ty::UnsafeBinder(inner) => {
            let generic_params = clean_bound_vars(inner.bound_vars());
            let ty = clean_middle_ty(inner.into(), cx, None, None);
            UnsafeBinder(Box::new(UnsafeBinderTy { generic_params, ty }))
        }
        ty::Adt(def, args) => {
            let did = def.did();
            let kind = match def.adt_kind() {
                AdtKind::Struct => ItemType::Struct,
                AdtKind::Union => ItemType::Union,
                AdtKind::Enum => ItemType::Enum,
            };
            inline::record_extern_fqn(cx, did, kind);
            let path = clean_middle_path(cx, did, false, ThinVec::new(), bound_ty.rebind(args));
            Type::Path { path }
        }
        ty::Foreign(did) => {
            inline::record_extern_fqn(cx, did, ItemType::ForeignType);
            let path = clean_middle_path(
                cx,
                did,
                false,
                ThinVec::new(),
                ty::Binder::dummy(ty::GenericArgs::empty()),
            );
            Type::Path { path }
        }
        ty::Dynamic(obj, ref reg, _) => {
            // HACK: pick the first `did` as the `did` of the trait object. Someone
            // might want to implement "native" support for marker-trait-only
            // trait objects.
            let mut dids = obj.auto_traits();
            let did = obj
                .principal_def_id()
                .or_else(|| dids.next())
                .unwrap_or_else(|| panic!("found trait object `{bound_ty:?}` with no traits?"));
            let args = match obj.principal() {
                Some(principal) => principal.map_bound(|p| p.args),
                // marker traits have no args.
                _ => ty::Binder::dummy(ty::GenericArgs::empty()),
            };

            inline::record_extern_fqn(cx, did, ItemType::Trait);

            let lifetime = clean_trait_object_lifetime_bound(*reg, container, obj, cx.tcx);

            let mut bounds = dids
                .map(|did| {
                    let empty = ty::Binder::dummy(ty::GenericArgs::empty());
                    let path = clean_middle_path(cx, did, false, ThinVec::new(), empty);
                    inline::record_extern_fqn(cx, did, ItemType::Trait);
                    PolyTrait { trait_: path, generic_params: Vec::new() }
                })
                .collect::<Vec<_>>();

            let constraints = obj
                .projection_bounds()
                .map(|pb| AssocItemConstraint {
                    assoc: projection_to_path_segment(
                        pb.map_bound(|pb| {
                            pb
                                // HACK(compiler-errors): Doesn't actually matter what self
                                // type we put here, because we're only using the GAT's args.
                                .with_self_ty(cx.tcx, cx.tcx.types.self_param)
                                .projection_term
                                // FIXME: This needs to be made resilient for `AliasTerm`s
                                // that are associated consts.
                                .expect_ty(cx.tcx)
                        }),
                        cx,
                    ),
                    kind: AssocItemConstraintKind::Equality {
                        term: clean_middle_term(pb.map_bound(|pb| pb.term), cx),
                    },
                })
                .collect();

            let late_bound_regions: FxIndexSet<_> = obj
                .iter()
                .flat_map(|pred| pred.bound_vars())
                .filter_map(|var| match var {
                    ty::BoundVariableKind::Region(ty::BoundRegionKind::Named(def_id, name))
                        if name != kw::UnderscoreLifetime =>
                    {
                        Some(GenericParamDef::lifetime(def_id, name))
                    }
                    _ => None,
                })
                .collect();
            let late_bound_regions = late_bound_regions.into_iter().collect();

            let path = clean_middle_path(cx, did, false, constraints, args);
            bounds.insert(0, PolyTrait { trait_: path, generic_params: late_bound_regions });

            DynTrait(bounds, lifetime)
        }
        ty::Tuple(t) => {
            Tuple(t.iter().map(|t| clean_middle_ty(bound_ty.rebind(t), cx, None, None)).collect())
        }

        ty::Alias(ty::Projection, data) => {
            clean_projection(bound_ty.rebind(data), cx, parent_def_id)
        }

        ty::Alias(ty::Inherent, alias_ty) => {
            let def_id = alias_ty.def_id;
            let alias_ty = bound_ty.rebind(alias_ty);
            let self_type = clean_middle_ty(alias_ty.map_bound(|ty| ty.self_ty()), cx, None, None);

            Type::QPath(Box::new(QPathData {
                assoc: PathSegment {
                    name: cx.tcx.associated_item(def_id).name,
                    args: GenericArgs::AngleBracketed {
                        args: clean_middle_generic_args(
                            cx,
                            alias_ty.map_bound(|ty| ty.args.as_slice()),
                            true,
                            def_id,
                        ),
                        constraints: Default::default(),
                    },
                },
                should_show_cast: false,
                self_type,
                trait_: None,
            }))
        }

        ty::Alias(ty::Weak, data) => {
            if cx.tcx.features().lazy_type_alias() {
                // Weak type alias `data` represents the `type X` in `type X = Y`. If we need `Y`,
                // we need to use `type_of`.
                let path = clean_middle_path(
                    cx,
                    data.def_id,
                    false,
                    ThinVec::new(),
                    bound_ty.rebind(data.args),
                );
                Type::Path { path }
            } else {
                let ty = cx.tcx.type_of(data.def_id).instantiate(cx.tcx, data.args);
                clean_middle_ty(bound_ty.rebind(ty), cx, None, None)
            }
        }

        ty::Param(ref p) => {
            if let Some(bounds) = cx.impl_trait_bounds.remove(&p.index.into()) {
                ImplTrait(bounds)
            } else if p.name == kw::SelfUpper {
                SelfTy
            } else {
                Generic(p.name)
            }
        }

        ty::Bound(_, ref ty) => match ty.kind {
            ty::BoundTyKind::Param(_, name) => Generic(name),
            ty::BoundTyKind::Anon => panic!("unexpected anonymous bound type variable"),
        },

        ty::Alias(ty::Opaque, ty::AliasTy { def_id, args, .. }) => {
            // If it's already in the same alias, don't get an infinite loop.
            if cx.current_type_aliases.contains_key(&def_id) {
                let path =
                    clean_middle_path(cx, def_id, false, ThinVec::new(), bound_ty.rebind(args));
                Type::Path { path }
            } else {
                *cx.current_type_aliases.entry(def_id).or_insert(0) += 1;
                // Grab the "TraitA + TraitB" from `impl TraitA + TraitB`,
                // by looking up the bounds associated with the def_id.
                let ty = clean_middle_opaque_bounds(cx, def_id, args);
                if let Some(count) = cx.current_type_aliases.get_mut(&def_id) {
                    *count -= 1;
                    if *count == 0 {
                        cx.current_type_aliases.remove(&def_id);
                    }
                }
                ty
            }
        }

        ty::Closure(..) => panic!("Closure"),
        ty::CoroutineClosure(..) => panic!("CoroutineClosure"),
        ty::Coroutine(..) => panic!("Coroutine"),
        ty::Placeholder(..) => panic!("Placeholder"),
        ty::CoroutineWitness(..) => panic!("CoroutineWitness"),
        ty::Infer(..) => panic!("Infer"),

        ty::Error(_) => FatalError.raise(),
    }
}

fn clean_middle_opaque_bounds<'tcx>(
    cx: &mut DocContext<'tcx>,
    impl_trait_def_id: DefId,
    args: ty::GenericArgsRef<'tcx>,
) -> Type {
    let mut has_sized = false;

    let bounds: Vec<_> = cx
        .tcx
        .explicit_item_bounds(impl_trait_def_id)
        .iter_instantiated_copied(cx.tcx, args)
        .collect();

    let mut bounds = bounds
        .iter()
        .filter_map(|(bound, _)| {
            let bound_predicate = bound.kind();
            let trait_ref = match bound_predicate.skip_binder() {
                ty::ClauseKind::Trait(tr) => bound_predicate.rebind(tr.trait_ref),
                ty::ClauseKind::TypeOutlives(ty::OutlivesPredicate(_ty, reg)) => {
                    return clean_middle_region(reg).map(GenericBound::Outlives);
                }
                _ => return None,
            };

            if let Some(sized) = cx.tcx.lang_items().sized_trait()
                && trait_ref.def_id() == sized
            {
                has_sized = true;
                return None;
            }

            let bindings: ThinVec<_> = bounds
                .iter()
                .filter_map(|(bound, _)| {
                    if let ty::ClauseKind::Projection(proj) = bound.kind().skip_binder()
                        && proj.projection_term.trait_ref(cx.tcx) == trait_ref.skip_binder()
                    {
                        return Some(AssocItemConstraint {
                            assoc: projection_to_path_segment(
                                // FIXME: This needs to be made resilient for `AliasTerm`s that
                                // are associated consts.
                                bound.kind().rebind(proj.projection_term.expect_ty(cx.tcx)),
                                cx,
                            ),
                            kind: AssocItemConstraintKind::Equality {
                                term: clean_middle_term(bound.kind().rebind(proj.term), cx),
                            },
                        });
                    }
                    None
                })
                .collect();

            Some(clean_poly_trait_ref_with_constraints(cx, trait_ref, bindings))
        })
        .collect::<Vec<_>>();

    if !has_sized {
        bounds.push(GenericBound::maybe_sized(cx));
    }

    // Move trait bounds to the front.
    bounds.sort_by_key(|b| !b.is_trait_bound());

    // Add back a `Sized` bound if there are no *trait* bounds remaining (incl. `?Sized`).
    // Since all potential trait bounds are at the front we can just check the first bound.
    if bounds.first().is_none_or(|b| !b.is_trait_bound()) {
        bounds.insert(0, GenericBound::sized(cx));
    }

    if let Some(args) = cx.tcx.rendered_precise_capturing_args(impl_trait_def_id) {
        bounds.push(GenericBound::Use(
            args.iter()
                .map(|arg| match arg {
                    hir::PreciseCapturingArgKind::Lifetime(lt) => {
                        PreciseCapturingArg::Lifetime(Lifetime(*lt))
                    }
                    hir::PreciseCapturingArgKind::Param(param) => {
                        PreciseCapturingArg::Param(*param)
                    }
                })
                .collect(),
        ));
    }

    ImplTrait(bounds)
}

pub(crate) fn clean_field<'tcx>(field: &hir::FieldDef<'tcx>, cx: &mut DocContext<'tcx>) -> Item {
    clean_field_with_def_id(field.def_id.to_def_id(), field.ident.name, clean_ty(field.ty, cx), cx)
}

pub(crate) fn clean_middle_field(field: &ty::FieldDef, cx: &mut DocContext<'_>) -> Item {
    clean_field_with_def_id(
        field.did,
        field.name,
        clean_middle_ty(
            ty::Binder::dummy(cx.tcx.type_of(field.did).instantiate_identity()),
            cx,
            Some(field.did),
            None,
        ),
        cx,
    )
}

pub(crate) fn clean_field_with_def_id(
    def_id: DefId,
    name: Symbol,
    ty: Type,
    cx: &mut DocContext<'_>,
) -> Item {
    Item::from_def_id_and_parts(def_id, Some(name), StructFieldItem(ty), cx)
}

pub(crate) fn clean_variant_def(variant: &ty::VariantDef, cx: &mut DocContext<'_>) -> Item {
    let discriminant = match variant.discr {
        ty::VariantDiscr::Explicit(def_id) => Some(Discriminant { expr: None, value: def_id }),
        ty::VariantDiscr::Relative(_) => None,
    };

    let kind = match variant.ctor_kind() {
        Some(CtorKind::Const) => VariantKind::CLike,
        Some(CtorKind::Fn) => VariantKind::Tuple(
            variant.fields.iter().map(|field| clean_middle_field(field, cx)).collect(),
        ),
        None => VariantKind::Struct(VariantStruct {
            fields: variant.fields.iter().map(|field| clean_middle_field(field, cx)).collect(),
        }),
    };

    Item::from_def_id_and_parts(
        variant.def_id,
        Some(variant.name),
        VariantItem(Variant { kind, discriminant }),
        cx,
    )
}

pub(crate) fn clean_variant_def_with_args<'tcx>(
    variant: &ty::VariantDef,
    args: &GenericArgsRef<'tcx>,
    cx: &mut DocContext<'tcx>,
) -> Item {
    let discriminant = match variant.discr {
        ty::VariantDiscr::Explicit(def_id) => Some(Discriminant { expr: None, value: def_id }),
        ty::VariantDiscr::Relative(_) => None,
    };

    use rustc_middle::traits::ObligationCause;
    use rustc_trait_selection::infer::TyCtxtInferExt;
    use rustc_trait_selection::traits::query::normalize::QueryNormalizeExt;

    let infcx = cx.tcx.infer_ctxt().build(TypingMode::non_body_analysis());
    let kind = match variant.ctor_kind() {
        Some(CtorKind::Const) => VariantKind::CLike,
        Some(CtorKind::Fn) => VariantKind::Tuple(
            variant
                .fields
                .iter()
                .map(|field| {
                    let ty = cx.tcx.type_of(field.did).instantiate(cx.tcx, args);

                    // normalize the type to only show concrete types
                    // note: we do not use try_normalize_erasing_regions since we
                    // do care about showing the regions
                    let ty = infcx
                        .at(&ObligationCause::dummy(), cx.param_env)
                        .query_normalize(ty)
                        .map(|normalized| normalized.value)
                        .unwrap_or(ty);

                    clean_field_with_def_id(
                        field.did,
                        field.name,
                        clean_middle_ty(ty::Binder::dummy(ty), cx, Some(field.did), None),
                        cx,
                    )
                })
                .collect(),
        ),
        None => VariantKind::Struct(VariantStruct {
            fields: variant
                .fields
                .iter()
                .map(|field| {
                    let ty = cx.tcx.type_of(field.did).instantiate(cx.tcx, args);

                    // normalize the type to only show concrete types
                    // note: we do not use try_normalize_erasing_regions since we
                    // do care about showing the regions
                    let ty = infcx
                        .at(&ObligationCause::dummy(), cx.param_env)
                        .query_normalize(ty)
                        .map(|normalized| normalized.value)
                        .unwrap_or(ty);

                    clean_field_with_def_id(
                        field.did,
                        field.name,
                        clean_middle_ty(ty::Binder::dummy(ty), cx, Some(field.did), None),
                        cx,
                    )
                })
                .collect(),
        }),
    };

    Item::from_def_id_and_parts(
        variant.def_id,
        Some(variant.name),
        VariantItem(Variant { kind, discriminant }),
        cx,
    )
}

fn clean_variant_data<'tcx>(
    variant: &hir::VariantData<'tcx>,
    disr_expr: &Option<&hir::AnonConst>,
    cx: &mut DocContext<'tcx>,
) -> Variant {
    let discriminant = disr_expr
        .map(|disr| Discriminant { expr: Some(disr.body), value: disr.def_id.to_def_id() });

    let kind = match variant {
        hir::VariantData::Struct { fields, .. } => VariantKind::Struct(VariantStruct {
            fields: fields.iter().map(|x| clean_field(x, cx)).collect(),
        }),
        hir::VariantData::Tuple(..) => {
            VariantKind::Tuple(variant.fields().iter().map(|x| clean_field(x, cx)).collect())
        }
        hir::VariantData::Unit(..) => VariantKind::CLike,
    };

    Variant { discriminant, kind }
}

fn clean_path<'tcx>(path: &hir::Path<'tcx>, cx: &mut DocContext<'tcx>) -> Path {
    Path {
        res: path.res,
        segments: path.segments.iter().map(|x| clean_path_segment(x, cx)).collect(),
    }
}

fn clean_generic_args<'tcx>(
    generic_args: &hir::GenericArgs<'tcx>,
    cx: &mut DocContext<'tcx>,
) -> GenericArgs {
    match generic_args.parenthesized {
        hir::GenericArgsParentheses::No => {
            let args = generic_args
                .args
                .iter()
                .map(|arg| match arg {
                    hir::GenericArg::Lifetime(lt) if !lt.is_anonymous() => {
                        GenericArg::Lifetime(clean_lifetime(lt, cx))
                    }
                    hir::GenericArg::Lifetime(_) => GenericArg::Lifetime(Lifetime::elided()),
                    hir::GenericArg::Type(ty) => GenericArg::Type(clean_ty(ty.as_unambig_ty(), cx)),
                    hir::GenericArg::Const(ct) => {
                        GenericArg::Const(Box::new(clean_const(ct.as_unambig_ct(), cx)))
                    }
                    hir::GenericArg::Infer(_inf) => GenericArg::Infer,
                })
                .collect();
            let constraints = generic_args
                .constraints
                .iter()
                .map(|c| clean_assoc_item_constraint(c, cx))
                .collect::<ThinVec<_>>();
            GenericArgs::AngleBracketed { args, constraints }
        }
        hir::GenericArgsParentheses::ParenSugar => {
            let Some((inputs, output)) = generic_args.paren_sugar_inputs_output() else {
                bug!();
            };
            let inputs = inputs.iter().map(|x| clean_ty(x, cx)).collect();
            let output = match output.kind {
                hir::TyKind::Tup(&[]) => None,
                _ => Some(Box::new(clean_ty(output, cx))),
            };
            GenericArgs::Parenthesized { inputs, output }
        }
        hir::GenericArgsParentheses::ReturnTypeNotation => GenericArgs::ReturnTypeNotation,
    }
}

fn clean_path_segment<'tcx>(
    path: &hir::PathSegment<'tcx>,
    cx: &mut DocContext<'tcx>,
) -> PathSegment {
    PathSegment { name: path.ident.name, args: clean_generic_args(path.args(), cx) }
}

fn clean_bare_fn_ty<'tcx>(
    bare_fn: &hir::BareFnTy<'tcx>,
    cx: &mut DocContext<'tcx>,
) -> BareFunctionDecl {
    let (generic_params, decl) = enter_impl_trait(cx, |cx| {
        // NOTE: generics must be cleaned before args
        let generic_params = bare_fn
            .generic_params
            .iter()
            .filter(|p| !is_elided_lifetime(p))
            .map(|x| clean_generic_param(cx, None, x))
            .collect();
        let args = clean_args_from_types_and_names(cx, bare_fn.decl.inputs, bare_fn.param_names);
        let decl = clean_fn_decl_with_args(cx, bare_fn.decl, None, args);
        (generic_params, decl)
    });
    BareFunctionDecl { safety: bare_fn.safety, abi: bare_fn.abi, decl, generic_params }
}

fn clean_unsafe_binder_ty<'tcx>(
    unsafe_binder_ty: &hir::UnsafeBinderTy<'tcx>,
    cx: &mut DocContext<'tcx>,
) -> UnsafeBinderTy {
    // NOTE: generics must be cleaned before args
    let generic_params = unsafe_binder_ty
        .generic_params
        .iter()
        .filter(|p| !is_elided_lifetime(p))
        .map(|x| clean_generic_param(cx, None, x))
        .collect();
    let ty = clean_ty(unsafe_binder_ty.inner_ty, cx);
    UnsafeBinderTy { generic_params, ty }
}

pub(crate) fn reexport_chain(
    tcx: TyCtxt<'_>,
    import_def_id: LocalDefId,
    target_def_id: DefId,
) -> &[Reexport] {
    for child in tcx.module_children_local(tcx.local_parent(import_def_id)) {
        if child.res.opt_def_id() == Some(target_def_id)
            && child.reexport_chain.first().and_then(|r| r.id()) == Some(import_def_id.to_def_id())
        {
            return &child.reexport_chain;
        }
    }
    &[]
}

/// Collect attributes from the whole import chain.
fn get_all_import_attributes<'hir>(
    cx: &mut DocContext<'hir>,
    import_def_id: LocalDefId,
    target_def_id: DefId,
    is_inline: bool,
) -> Vec<(Cow<'hir, hir::Attribute>, Option<DefId>)> {
    let mut attrs = Vec::new();
    let mut first = true;
    for def_id in reexport_chain(cx.tcx, import_def_id, target_def_id)
        .iter()
        .flat_map(|reexport| reexport.id())
    {
        let import_attrs = inline::load_attrs(cx, def_id);
        if first {
            // This is the "original" reexport so we get all its attributes without filtering them.
            attrs = import_attrs.iter().map(|attr| (Cow::Borrowed(attr), Some(def_id))).collect();
            first = false;
        // We don't add attributes of an intermediate re-export if it has `#[doc(hidden)]`.
        } else if cx.render_options.document_hidden || !cx.tcx.is_doc_hidden(def_id) {
            add_without_unwanted_attributes(&mut attrs, import_attrs, is_inline, Some(def_id));
        }
    }
    attrs
}

fn filter_tokens_from_list(
    args_tokens: &TokenStream,
    should_retain: impl Fn(&TokenTree) -> bool,
) -> Vec<TokenTree> {
    let mut tokens = Vec::with_capacity(args_tokens.len());
    let mut skip_next_comma = false;
    for token in args_tokens.iter() {
        match token {
            TokenTree::Token(Token { kind: TokenKind::Comma, .. }, _) if skip_next_comma => {
                skip_next_comma = false;
            }
            token if should_retain(token) => {
                skip_next_comma = false;
                tokens.push(token.clone());
            }
            _ => {
                skip_next_comma = true;
            }
        }
    }
    tokens
}

fn filter_doc_attr_ident(ident: Symbol, is_inline: bool) -> bool {
    if is_inline {
        ident == sym::hidden || ident == sym::inline || ident == sym::no_inline
    } else {
        ident == sym::cfg
    }
}

/// Remove attributes from `normal` that should not be inherited by `use` re-export.
/// Before calling this function, make sure `normal` is a `#[doc]` attribute.
fn filter_doc_attr(args: &mut hir::AttrArgs, is_inline: bool) {
    match args {
        hir::AttrArgs::Delimited(args) => {
            let tokens = filter_tokens_from_list(&args.tokens, |token| {
                !matches!(
                    token,
                    TokenTree::Token(
                        Token {
                            kind: TokenKind::Ident(
                                ident,
                                _,
                            ),
                            ..
                        },
                        _,
                    ) if filter_doc_attr_ident(*ident, is_inline),
                )
            });
            args.tokens = TokenStream::new(tokens);
        }
        hir::AttrArgs::Empty | hir::AttrArgs::Eq { .. } => {}
    }
}

/// When inlining items, we merge their attributes (and all the reexports attributes too) with the
/// final reexport. For example:
///
/// ```ignore (just an example)
/// #[doc(hidden, cfg(feature = "foo"))]
/// pub struct Foo;
///
/// #[doc(cfg(feature = "bar"))]
/// #[doc(hidden, no_inline)]
/// pub use Foo as Foo1;
///
/// #[doc(inline)]
/// pub use Foo2 as Bar;
/// ```
///
/// So `Bar` at the end will have both `cfg(feature = "...")`. However, we don't want to merge all
/// attributes so we filter out the following ones:
/// * `doc(inline)`
/// * `doc(no_inline)`
/// * `doc(hidden)`
fn add_without_unwanted_attributes<'hir>(
    attrs: &mut Vec<(Cow<'hir, hir::Attribute>, Option<DefId>)>,
    new_attrs: &'hir [hir::Attribute],
    is_inline: bool,
    import_parent: Option<DefId>,
) {
    for attr in new_attrs {
        if attr.is_doc_comment() {
            attrs.push((Cow::Borrowed(attr), import_parent));
            continue;
        }
        let mut attr = attr.clone();
        match attr {
            hir::Attribute::Unparsed(ref mut normal) if let [ident] = &*normal.path.segments => {
                let ident = ident.name;
                if ident == sym::doc {
                    filter_doc_attr(&mut normal.args, is_inline);
                    attrs.push((Cow::Owned(attr), import_parent));
                } else if is_inline || ident != sym::cfg_trace {
                    // If it's not a `cfg()` attribute, we keep it.
                    attrs.push((Cow::Owned(attr), import_parent));
                }
            }
            hir::Attribute::Parsed(..) if is_inline => {
                attrs.push((Cow::Owned(attr), import_parent));
            }
            _ => {}
        }
    }
}

fn clean_maybe_renamed_item<'tcx>(
    cx: &mut DocContext<'tcx>,
    item: &hir::Item<'tcx>,
    renamed: Option<Symbol>,
    import_id: Option<LocalDefId>,
) -> Vec<Item> {
    use hir::ItemKind;

    let def_id = item.owner_id.to_def_id();
    let mut name = renamed.unwrap_or_else(|| {
        // FIXME: using kw::Empty is a bit of a hack
        cx.tcx.hir_opt_name(item.hir_id()).unwrap_or(kw::Empty)
    });

    cx.with_param_env(def_id, |cx| {
        let kind = match item.kind {
            ItemKind::Static(_, ty, mutability, body_id) => StaticItem(Static {
                type_: Box::new(clean_ty(ty, cx)),
                mutability,
                expr: Some(body_id),
            }),
            ItemKind::Const(_, ty, generics, body_id) => ConstantItem(Box::new(Constant {
                generics: clean_generics(generics, cx),
                type_: clean_ty(ty, cx),
                kind: ConstantKind::Local { body: body_id, def_id },
            })),
            ItemKind::TyAlias(_, hir_ty, generics) => {
                *cx.current_type_aliases.entry(def_id).or_insert(0) += 1;
                let rustdoc_ty = clean_ty(hir_ty, cx);
                let type_ =
                    clean_middle_ty(ty::Binder::dummy(lower_ty(cx.tcx, hir_ty)), cx, None, None);
                let generics = clean_generics(generics, cx);
                if let Some(count) = cx.current_type_aliases.get_mut(&def_id) {
                    *count -= 1;
                    if *count == 0 {
                        cx.current_type_aliases.remove(&def_id);
                    }
                }

                let ty = cx.tcx.type_of(def_id).instantiate_identity();

                let mut ret = Vec::new();
                let inner_type = clean_ty_alias_inner_type(ty, cx, &mut ret);

                ret.push(generate_item_with_correct_attrs(
                    cx,
                    TypeAliasItem(Box::new(TypeAlias {
                        generics,
                        inner_type,
                        type_: rustdoc_ty,
                        item_type: Some(type_),
                    })),
                    item.owner_id.def_id.to_def_id(),
                    name,
                    import_id,
                    renamed,
                ));
                return ret;
            }
            ItemKind::Enum(_, ref def, generics) => EnumItem(Enum {
                variants: def.variants.iter().map(|v| clean_variant(v, cx)).collect(),
                generics: clean_generics(generics, cx),
            }),
            ItemKind::TraitAlias(_, generics, bounds) => TraitAliasItem(TraitAlias {
                generics: clean_generics(generics, cx),
                bounds: bounds.iter().filter_map(|x| clean_generic_bound(x, cx)).collect(),
            }),
            ItemKind::Union(_, ref variant_data, generics) => UnionItem(Union {
                generics: clean_generics(generics, cx),
                fields: variant_data.fields().iter().map(|x| clean_field(x, cx)).collect(),
            }),
            ItemKind::Struct(_, ref variant_data, generics) => StructItem(Struct {
                ctor_kind: variant_data.ctor_kind(),
                generics: clean_generics(generics, cx),
                fields: variant_data.fields().iter().map(|x| clean_field(x, cx)).collect(),
            }),
            ItemKind::Impl(impl_) => return clean_impl(impl_, item.owner_id.def_id, cx),
            ItemKind::Macro(_, macro_def, MacroKind::Bang) => MacroItem(Macro {
                source: display_macro_source(cx, name, macro_def),
                macro_rules: macro_def.macro_rules,
            }),
            ItemKind::Macro(_, _, macro_kind) => clean_proc_macro(item, &mut name, macro_kind, cx),
            // proc macros can have a name set by attributes
            ItemKind::Fn { ref sig, generics, body: body_id, .. } => {
                clean_fn_or_proc_macro(item, sig, generics, body_id, &mut name, cx)
            }
            ItemKind::Trait(_, _, _, generics, bounds, item_ids) => {
                let items = item_ids
                    .iter()
                    .map(|ti| clean_trait_item(cx.tcx.hir_trait_item(ti.id), cx))
                    .collect();

                TraitItem(Box::new(Trait {
                    def_id,
                    items,
                    generics: clean_generics(generics, cx),
                    bounds: bounds.iter().filter_map(|x| clean_generic_bound(x, cx)).collect(),
                }))
            }
            ItemKind::ExternCrate(orig_name, _) => {
                return clean_extern_crate(item, name, orig_name, cx);
            }
            ItemKind::Use(path, kind) => {
                return clean_use_statement(item, name, path, kind, cx, &mut FxHashSet::default());
            }
            _ => span_bug!(item.span, "not yet converted"),
        };

        vec![generate_item_with_correct_attrs(
            cx,
            kind,
            item.owner_id.def_id.to_def_id(),
            name,
            import_id,
            renamed,
        )]
    })
}

fn clean_variant<'tcx>(variant: &hir::Variant<'tcx>, cx: &mut DocContext<'tcx>) -> Item {
    let kind = VariantItem(clean_variant_data(&variant.data, &variant.disr_expr, cx));
    Item::from_def_id_and_parts(variant.def_id.to_def_id(), Some(variant.ident.name), kind, cx)
}

fn clean_impl<'tcx>(
    impl_: &hir::Impl<'tcx>,
    def_id: LocalDefId,
    cx: &mut DocContext<'tcx>,
) -> Vec<Item> {
    let tcx = cx.tcx;
    let mut ret = Vec::new();
    let trait_ = impl_.of_trait.as_ref().map(|t| clean_trait_ref(t, cx));
    let items = impl_
        .items
        .iter()
        .map(|ii| clean_impl_item(tcx.hir_impl_item(ii.id), cx))
        .collect::<Vec<_>>();

    // If this impl block is an implementation of the Deref trait, then we
    // need to try inlining the target's inherent impl blocks as well.
    if trait_.as_ref().map(|t| t.def_id()) == tcx.lang_items().deref_trait() {
        build_deref_target_impls(cx, &items, &mut ret);
    }

    let for_ = clean_ty(impl_.self_ty, cx);
    let type_alias =
        for_.def_id(&cx.cache).and_then(|alias_def_id: DefId| match tcx.def_kind(alias_def_id) {
            DefKind::TyAlias => Some(clean_middle_ty(
                ty::Binder::dummy(tcx.type_of(def_id).instantiate_identity()),
                cx,
                Some(def_id.to_def_id()),
                None,
            )),
            _ => None,
        });
    let mut make_item = |trait_: Option<Path>, for_: Type, items: Vec<Item>| {
        let kind = ImplItem(Box::new(Impl {
            safety: impl_.safety,
            generics: clean_generics(impl_.generics, cx),
            trait_,
            for_,
            items,
            polarity: tcx.impl_polarity(def_id),
            kind: if utils::has_doc_flag(tcx, def_id.to_def_id(), sym::fake_variadic) {
                ImplKind::FakeVariadic
            } else {
                ImplKind::Normal
            },
        }));
        Item::from_def_id_and_parts(def_id.to_def_id(), None, kind, cx)
    };
    if let Some(type_alias) = type_alias {
        ret.push(make_item(trait_.clone(), type_alias, items.clone()));
    }
    ret.push(make_item(trait_, for_, items));
    ret
}

fn clean_extern_crate<'tcx>(
    krate: &hir::Item<'tcx>,
    name: Symbol,
    orig_name: Option<Symbol>,
    cx: &mut DocContext<'tcx>,
) -> Vec<Item> {
    // this is the ID of the `extern crate` statement
    let cnum = cx.tcx.extern_mod_stmt_cnum(krate.owner_id.def_id).unwrap_or(LOCAL_CRATE);
    // this is the ID of the crate itself
    let crate_def_id = cnum.as_def_id();
    let attrs = cx.tcx.hir_attrs(krate.hir_id());
    let ty_vis = cx.tcx.visibility(krate.owner_id);
    let please_inline = ty_vis.is_public()
        && attrs.iter().any(|a| {
            a.has_name(sym::doc)
                && match a.meta_item_list() {
                    Some(l) => ast::attr::list_contains_name(&l, sym::inline),
                    None => false,
                }
        })
        && !cx.is_json_output();

    let krate_owner_def_id = krate.owner_id.def_id;
    if please_inline
        && let Some(items) = inline::try_inline(
            cx,
            Res::Def(DefKind::Mod, crate_def_id),
            name,
            Some((attrs, Some(krate_owner_def_id))),
            &mut Default::default(),
        )
    {
        return items;
    }

    vec![Item::from_def_id_and_parts(
        krate_owner_def_id.to_def_id(),
        Some(name),
        ExternCrateItem { src: orig_name },
        cx,
    )]
}

fn clean_use_statement<'tcx>(
    import: &hir::Item<'tcx>,
    name: Symbol,
    path: &hir::UsePath<'tcx>,
    kind: hir::UseKind,
    cx: &mut DocContext<'tcx>,
    inlined_names: &mut FxHashSet<(ItemType, Symbol)>,
) -> Vec<Item> {
    let mut items = Vec::new();
    let hir::UsePath { segments, ref res, span } = *path;
    for &res in res {
        let path = hir::Path { segments, res, span };
        items.append(&mut clean_use_statement_inner(import, name, &path, kind, cx, inlined_names));
    }
    items
}

fn clean_use_statement_inner<'tcx>(
    import: &hir::Item<'tcx>,
    name: Symbol,
    path: &hir::Path<'tcx>,
    kind: hir::UseKind,
    cx: &mut DocContext<'tcx>,
    inlined_names: &mut FxHashSet<(ItemType, Symbol)>,
) -> Vec<Item> {
    if should_ignore_res(path.res) {
        return Vec::new();
    }
    // We need this comparison because some imports (for std types for example)
    // are "inserted" as well but directly by the compiler and they should not be
    // taken into account.
    if import.span.ctxt().outer_expn_data().kind == ExpnKind::AstPass(AstPass::StdImports) {
        return Vec::new();
    }

    let visibility = cx.tcx.visibility(import.owner_id);
    let attrs = cx.tcx.hir_attrs(import.hir_id());
    let inline_attr = hir_attr_lists(attrs, sym::doc).get_word_attr(sym::inline);
    let pub_underscore = visibility.is_public() && name == kw::Underscore;
    let current_mod = cx.tcx.parent_module_from_def_id(import.owner_id.def_id);
    let import_def_id = import.owner_id.def_id;

    // The parent of the module in which this import resides. This
    // is the same as `current_mod` if that's already the top
    // level module.
    let parent_mod = cx.tcx.parent_module_from_def_id(current_mod.to_local_def_id());

    // This checks if the import can be seen from a higher level module.
    // In other words, it checks if the visibility is the equivalent of
    // `pub(super)` or higher. If the current module is the top level
    // module, there isn't really a parent module, which makes the results
    // meaningless. In this case, we make sure the answer is `false`.
    let is_visible_from_parent_mod =
        visibility.is_accessible_from(parent_mod, cx.tcx) && !current_mod.is_top_level_module();

    if pub_underscore && let Some(ref inline) = inline_attr {
        struct_span_code_err!(
            cx.tcx.dcx(),
            inline.span(),
            E0780,
            "anonymous imports cannot be inlined"
        )
        .with_span_label(import.span, "anonymous import")
        .emit();
    }

    // We consider inlining the documentation of `pub use` statements, but we
    // forcefully don't inline if this is not public or if the
    // #[doc(no_inline)] attribute is present.
    // Don't inline doc(hidden) imports so they can be stripped at a later stage.
    let mut denied = cx.is_json_output()
        || !(visibility.is_public()
            || (cx.render_options.document_private && is_visible_from_parent_mod))
        || pub_underscore
        || attrs.iter().any(|a| {
            a.has_name(sym::doc)
                && match a.meta_item_list() {
                    Some(l) => {
                        ast::attr::list_contains_name(&l, sym::no_inline)
                            || ast::attr::list_contains_name(&l, sym::hidden)
                    }
                    None => false,
                }
        });

    // Also check whether imports were asked to be inlined, in case we're trying to re-export a
    // crate in Rust 2018+
    let path = clean_path(path, cx);
    let inner = if kind == hir::UseKind::Glob {
        if !denied {
            let mut visited = DefIdSet::default();
            if let Some(items) = inline::try_inline_glob(
                cx,
                path.res,
                current_mod,
                &mut visited,
                inlined_names,
                import,
            ) {
                return items;
            }
        }
        Import::new_glob(resolve_use_source(cx, path), true)
    } else {
        if inline_attr.is_none()
            && let Res::Def(DefKind::Mod, did) = path.res
            && !did.is_local()
            && did.is_crate_root()
        {
            // if we're `pub use`ing an extern crate root, don't inline it unless we
            // were specifically asked for it
            denied = true;
        }
        if !denied
            && let Some(mut items) = inline::try_inline(
                cx,
                path.res,
                name,
                Some((attrs, Some(import_def_id))),
                &mut Default::default(),
            )
        {
            items.push(Item::from_def_id_and_parts(
                import_def_id.to_def_id(),
                None,
                ImportItem(Import::new_simple(name, resolve_use_source(cx, path), false)),
                cx,
            ));
            return items;
        }
        Import::new_simple(name, resolve_use_source(cx, path), true)
    };

    vec![Item::from_def_id_and_parts(import_def_id.to_def_id(), None, ImportItem(inner), cx)]
}

fn clean_maybe_renamed_foreign_item<'tcx>(
    cx: &mut DocContext<'tcx>,
    item: &hir::ForeignItem<'tcx>,
    renamed: Option<Symbol>,
) -> Item {
    let def_id = item.owner_id.to_def_id();
    cx.with_param_env(def_id, |cx| {
        let kind = match item.kind {
            hir::ForeignItemKind::Fn(sig, names, generics) => ForeignFunctionItem(
                clean_function(cx, &sig, generics, FunctionArgs::Names(names)),
                sig.header.safety(),
            ),
            hir::ForeignItemKind::Static(ty, mutability, safety) => ForeignStaticItem(
                Static { type_: Box::new(clean_ty(ty, cx)), mutability, expr: None },
                safety,
            ),
            hir::ForeignItemKind::Type => ForeignTypeItem,
        };

        Item::from_def_id_and_parts(
            item.owner_id.def_id.to_def_id(),
            Some(renamed.unwrap_or(item.ident.name)),
            kind,
            cx,
        )
    })
}

fn clean_assoc_item_constraint<'tcx>(
    constraint: &hir::AssocItemConstraint<'tcx>,
    cx: &mut DocContext<'tcx>,
) -> AssocItemConstraint {
    AssocItemConstraint {
        assoc: PathSegment {
            name: constraint.ident.name,
            args: clean_generic_args(constraint.gen_args, cx),
        },
        kind: match constraint.kind {
            hir::AssocItemConstraintKind::Equality { ref term } => {
                AssocItemConstraintKind::Equality { term: clean_hir_term(term, cx) }
            }
            hir::AssocItemConstraintKind::Bound { bounds } => AssocItemConstraintKind::Bound {
                bounds: bounds.iter().filter_map(|b| clean_generic_bound(b, cx)).collect(),
            },
        },
    }
}

fn clean_bound_vars(bound_vars: &ty::List<ty::BoundVariableKind>) -> Vec<GenericParamDef> {
    bound_vars
        .into_iter()
        .filter_map(|var| match var {
            ty::BoundVariableKind::Region(ty::BoundRegionKind::Named(def_id, name))
                if name != kw::UnderscoreLifetime =>
            {
                Some(GenericParamDef::lifetime(def_id, name))
            }
            ty::BoundVariableKind::Ty(ty::BoundTyKind::Param(def_id, name)) => {
                Some(GenericParamDef {
                    name,
                    def_id,
                    kind: GenericParamDefKind::Type {
                        bounds: ThinVec::new(),
                        default: None,
                        synthetic: false,
                    },
                })
            }
            // FIXME(non_lifetime_binders): Support higher-ranked const parameters.
            ty::BoundVariableKind::Const => None,
            _ => None,
        })
        .collect()
}
