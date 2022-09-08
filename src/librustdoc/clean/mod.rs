//! This module contains the "cleaned" pieces of the AST, and the functions
//! that clean them.

mod auto_trait;
mod blanket_impl;
pub(crate) mod cfg;
pub(crate) mod inline;
mod render_macro_matchers;
mod simplify;
pub(crate) mod types;
pub(crate) mod utils;

use rustc_ast as ast;
use rustc_attr as attr;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_hir as hir;
use rustc_hir::def::{CtorKind, DefKind, Res};
use rustc_hir::def_id::{DefId, LOCAL_CRATE};
use rustc_hir::PredicateOrigin;
use rustc_infer::infer::region_constraints::{Constraint, RegionConstraintData};
use rustc_middle::middle::resolve_lifetime as rl;
use rustc_middle::ty::fold::TypeFolder;
use rustc_middle::ty::subst::{InternalSubsts, Subst};
use rustc_middle::ty::{self, AdtKind, DefIdTree, EarlyBinder, Lift, Ty, TyCtxt};
use rustc_middle::{bug, span_bug};
use rustc_span::hygiene::{AstPass, MacroKind};
use rustc_span::symbol::{kw, sym, Ident, Symbol};
use rustc_span::{self, ExpnKind};
use rustc_typeck::hir_ty_to_ty;

use std::assert_matches::assert_matches;
use std::collections::hash_map::Entry;
use std::collections::BTreeMap;
use std::default::Default;
use std::hash::Hash;
use std::mem;
use thin_vec::ThinVec;

use crate::core::{self, DocContext, ImplTraitParam};
use crate::formats::item_type::ItemType;
use crate::visit_ast::Module as DocModule;

use utils::*;

pub(crate) use self::types::*;
pub(crate) use self::utils::{get_auto_trait_and_blanket_impls, krate, register_res};

pub(crate) fn clean_doc_module<'tcx>(doc: &DocModule<'tcx>, cx: &mut DocContext<'tcx>) -> Item {
    let mut items: Vec<Item> = vec![];
    let mut inserted = FxHashSet::default();
    items.extend(doc.foreigns.iter().map(|(item, renamed)| {
        let item = clean_maybe_renamed_foreign_item(cx, item, *renamed);
        if let Some(name) = item.name {
            inserted.insert((item.type_(), name));
        }
        item
    }));
    items.extend(doc.mods.iter().map(|x| {
        inserted.insert((ItemType::Module, x.name));
        clean_doc_module(x, cx)
    }));

    // Split up imports from all other items.
    //
    // This covers the case where somebody does an import which should pull in an item,
    // but there's already an item with the same namespace and same name. Rust gives
    // priority to the not-imported one, so we should, too.
    items.extend(doc.items.iter().flat_map(|(item, renamed)| {
        // First, lower everything other than imports.
        if matches!(item.kind, hir::ItemKind::Use(_, hir::UseKind::Glob)) {
            return Vec::new();
        }
        let v = clean_maybe_renamed_item(cx, item, *renamed);
        for item in &v {
            if let Some(name) = item.name {
                inserted.insert((item.type_(), name));
            }
        }
        v
    }));
    items.extend(doc.items.iter().flat_map(|(item, renamed)| {
        // Now we actually lower the imports, skipping everything else.
        if let hir::ItemKind::Use(path, hir::UseKind::Glob) = item.kind {
            let name = renamed.unwrap_or_else(|| cx.tcx.hir().name(item.hir_id()));
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

    Item::from_hir_id_and_parts(doc.id, Some(doc.name), ModuleItem(Module { items, span }), cx)
}

fn clean_generic_bound<'tcx>(
    bound: &hir::GenericBound<'tcx>,
    cx: &mut DocContext<'tcx>,
) -> Option<GenericBound> {
    Some(match *bound {
        hir::GenericBound::Outlives(lt) => GenericBound::Outlives(clean_lifetime(lt, cx)),
        hir::GenericBound::LangItemTrait(lang_item, span, _, generic_args) => {
            let def_id = cx.tcx.require_lang_item(lang_item, Some(span));

            let trait_ref = ty::TraitRef::identity(cx.tcx, def_id).skip_binder();

            let generic_args = clean_generic_args(generic_args, cx);
            let GenericArgs::AngleBracketed { bindings, .. } = generic_args
            else {
                bug!("clean: parenthesized `GenericBound::LangItemTrait`");
            };

            let trait_ = clean_trait_ref_with_bindings(cx, trait_ref, bindings);
            GenericBound::TraitBound(
                PolyTrait { trait_, generic_params: vec![] },
                hir::TraitBoundModifier::None,
            )
        }
        hir::GenericBound::Trait(ref t, modifier) => {
            // `T: ~const Destruct` is hidden because `T: Destruct` is a no-op.
            if modifier == hir::TraitBoundModifier::MaybeConst
                && cx.tcx.lang_items().destruct_trait() == Some(t.trait_ref.trait_def_id().unwrap())
            {
                return None;
            }

            GenericBound::TraitBound(clean_poly_trait_ref(t, cx), modifier)
        }
    })
}

pub(crate) fn clean_trait_ref_with_bindings<'tcx>(
    cx: &mut DocContext<'tcx>,
    trait_ref: ty::TraitRef<'tcx>,
    bindings: ThinVec<TypeBinding>,
) -> Path {
    let kind = cx.tcx.def_kind(trait_ref.def_id).into();
    if !matches!(kind, ItemType::Trait | ItemType::TraitAlias) {
        span_bug!(cx.tcx.def_span(trait_ref.def_id), "`TraitRef` had unexpected kind {:?}", kind);
    }
    inline::record_extern_fqn(cx, trait_ref.def_id, kind);
    let path = external_path(cx, trait_ref.def_id, true, bindings, trait_ref.substs);

    debug!("ty::TraitRef\n  subst: {:?}\n", trait_ref.substs);

    path
}

fn clean_poly_trait_ref_with_bindings<'tcx>(
    cx: &mut DocContext<'tcx>,
    poly_trait_ref: ty::PolyTraitRef<'tcx>,
    bindings: ThinVec<TypeBinding>,
) -> GenericBound {
    let poly_trait_ref = poly_trait_ref.lift_to_tcx(cx.tcx).unwrap();

    // collect any late bound regions
    let late_bound_regions: Vec<_> = cx
        .tcx
        .collect_referenced_late_bound_regions(&poly_trait_ref)
        .into_iter()
        .filter_map(|br| match br {
            ty::BrNamed(_, name) if name != kw::UnderscoreLifetime => Some(GenericParamDef {
                name,
                kind: GenericParamDefKind::Lifetime { outlives: vec![] },
            }),
            _ => None,
        })
        .collect();

    let trait_ = clean_trait_ref_with_bindings(cx, poly_trait_ref.skip_binder(), bindings);
    GenericBound::TraitBound(
        PolyTrait { trait_, generic_params: late_bound_regions },
        hir::TraitBoundModifier::None,
    )
}

fn clean_lifetime<'tcx>(lifetime: &hir::Lifetime, cx: &mut DocContext<'tcx>) -> Lifetime {
    let def = cx.tcx.named_region(lifetime.hir_id);
    if let Some(
        rl::Region::EarlyBound(node_id)
        | rl::Region::LateBound(_, _, node_id)
        | rl::Region::Free(_, node_id),
    ) = def
    {
        if let Some(lt) = cx.substs.get(&node_id).and_then(|p| p.as_lt()).cloned() {
            return lt;
        }
    }
    Lifetime(lifetime.name.ident().name)
}

pub(crate) fn clean_const<'tcx>(constant: &hir::ConstArg, cx: &mut DocContext<'tcx>) -> Constant {
    let def_id = cx.tcx.hir().body_owner_def_id(constant.value.body).to_def_id();
    Constant {
        type_: clean_middle_ty(cx.tcx.type_of(def_id), cx, Some(def_id)),
        kind: ConstantKind::Anonymous { body: constant.value.body },
    }
}

pub(crate) fn clean_middle_const<'tcx>(
    constant: ty::Const<'tcx>,
    cx: &mut DocContext<'tcx>,
) -> Constant {
    // FIXME: instead of storing the stringified expression, store `self` directly instead.
    Constant {
        type_: clean_middle_ty(constant.ty(), cx, None),
        kind: ConstantKind::TyConst { expr: constant.to_string() },
    }
}

pub(crate) fn clean_middle_region<'tcx>(region: ty::Region<'tcx>) -> Option<Lifetime> {
    match *region {
        ty::ReStatic => Some(Lifetime::statik()),
        ty::ReLateBound(_, ty::BoundRegion { kind: ty::BrNamed(_, name), .. }) => {
            if name != kw::UnderscoreLifetime { Some(Lifetime(name)) } else { None }
        }
        ty::ReEarlyBound(ref data) => {
            if data.name != kw::UnderscoreLifetime {
                Some(Lifetime(data.name))
            } else {
                None
            }
        }
        ty::ReLateBound(..)
        | ty::ReFree(..)
        | ty::ReVar(..)
        | ty::RePlaceholder(..)
        | ty::ReEmpty(_)
        | ty::ReErased => {
            debug!("cannot clean region {:?}", region);
            None
        }
    }
}

fn clean_where_predicate<'tcx>(
    predicate: &hir::WherePredicate<'tcx>,
    cx: &mut DocContext<'tcx>,
) -> Option<WherePredicate> {
    if !predicate.in_where_clause() {
        return None;
    }
    Some(match *predicate {
        hir::WherePredicate::BoundPredicate(ref wbp) => {
            let bound_params = wbp
                .bound_generic_params
                .iter()
                .map(|param| {
                    // Higher-ranked params must be lifetimes.
                    // Higher-ranked lifetimes can't have bounds.
                    assert_matches!(
                        param,
                        hir::GenericParam { kind: hir::GenericParamKind::Lifetime { .. }, .. }
                    );
                    Lifetime(param.name.ident().name)
                })
                .collect();
            WherePredicate::BoundPredicate {
                ty: clean_ty(wbp.bounded_ty, cx),
                bounds: wbp.bounds.iter().filter_map(|x| clean_generic_bound(x, cx)).collect(),
                bound_params,
            }
        }

        hir::WherePredicate::RegionPredicate(ref wrp) => WherePredicate::RegionPredicate {
            lifetime: clean_lifetime(wrp.lifetime, cx),
            bounds: wrp.bounds.iter().filter_map(|x| clean_generic_bound(x, cx)).collect(),
        },

        hir::WherePredicate::EqPredicate(ref wrp) => WherePredicate::EqPredicate {
            lhs: clean_ty(wrp.lhs_ty, cx),
            rhs: clean_ty(wrp.rhs_ty, cx).into(),
        },
    })
}

pub(crate) fn clean_predicate<'tcx>(
    predicate: ty::Predicate<'tcx>,
    cx: &mut DocContext<'tcx>,
) -> Option<WherePredicate> {
    let bound_predicate = predicate.kind();
    match bound_predicate.skip_binder() {
        ty::PredicateKind::Trait(pred) => {
            clean_poly_trait_predicate(bound_predicate.rebind(pred), cx)
        }
        ty::PredicateKind::RegionOutlives(pred) => clean_region_outlives_predicate(pred),
        ty::PredicateKind::TypeOutlives(pred) => clean_type_outlives_predicate(pred, cx),
        ty::PredicateKind::Projection(pred) => Some(clean_projection_predicate(pred, cx)),
        ty::PredicateKind::ConstEvaluatable(..) => None,
        ty::PredicateKind::WellFormed(..) => None,

        ty::PredicateKind::Subtype(..)
        | ty::PredicateKind::Coerce(..)
        | ty::PredicateKind::ObjectSafe(..)
        | ty::PredicateKind::ClosureKind(..)
        | ty::PredicateKind::ConstEquate(..)
        | ty::PredicateKind::TypeWellFormedFromEnv(..) => panic!("not user writable"),
    }
}

fn clean_poly_trait_predicate<'tcx>(
    pred: ty::PolyTraitPredicate<'tcx>,
    cx: &mut DocContext<'tcx>,
) -> Option<WherePredicate> {
    // `T: ~const Destruct` is hidden because `T: Destruct` is a no-op.
    if pred.skip_binder().constness == ty::BoundConstness::ConstIfConst
        && Some(pred.skip_binder().def_id()) == cx.tcx.lang_items().destruct_trait()
    {
        return None;
    }

    let poly_trait_ref = pred.map_bound(|pred| pred.trait_ref);
    Some(WherePredicate::BoundPredicate {
        ty: clean_middle_ty(poly_trait_ref.skip_binder().self_ty(), cx, None),
        bounds: vec![clean_poly_trait_ref_with_bindings(cx, poly_trait_ref, ThinVec::new())],
        bound_params: Vec::new(),
    })
}

fn clean_region_outlives_predicate<'tcx>(
    pred: ty::OutlivesPredicate<ty::Region<'tcx>, ty::Region<'tcx>>,
) -> Option<WherePredicate> {
    let ty::OutlivesPredicate(a, b) = pred;

    if a.is_empty() && b.is_empty() {
        return None;
    }

    Some(WherePredicate::RegionPredicate {
        lifetime: clean_middle_region(a).expect("failed to clean lifetime"),
        bounds: vec![GenericBound::Outlives(
            clean_middle_region(b).expect("failed to clean bounds"),
        )],
    })
}

fn clean_type_outlives_predicate<'tcx>(
    pred: ty::OutlivesPredicate<Ty<'tcx>, ty::Region<'tcx>>,
    cx: &mut DocContext<'tcx>,
) -> Option<WherePredicate> {
    let ty::OutlivesPredicate(ty, lt) = pred;

    if lt.is_empty() {
        return None;
    }

    Some(WherePredicate::BoundPredicate {
        ty: clean_middle_ty(ty, cx, None),
        bounds: vec![GenericBound::Outlives(
            clean_middle_region(lt).expect("failed to clean lifetimes"),
        )],
        bound_params: Vec::new(),
    })
}

fn clean_middle_term<'tcx>(term: ty::Term<'tcx>, cx: &mut DocContext<'tcx>) -> Term {
    match term.unpack() {
        ty::TermKind::Ty(ty) => Term::Type(clean_middle_ty(ty, cx, None)),
        ty::TermKind::Const(c) => Term::Constant(clean_middle_const(c, cx)),
    }
}

fn clean_hir_term<'tcx>(term: &hir::Term<'tcx>, cx: &mut DocContext<'tcx>) -> Term {
    match term {
        hir::Term::Ty(ty) => Term::Type(clean_ty(ty, cx)),
        hir::Term::Const(c) => {
            let def_id = cx.tcx.hir().local_def_id(c.hir_id);
            Term::Constant(clean_middle_const(ty::Const::from_anon_const(cx.tcx, def_id), cx))
        }
    }
}

fn clean_projection_predicate<'tcx>(
    pred: ty::ProjectionPredicate<'tcx>,
    cx: &mut DocContext<'tcx>,
) -> WherePredicate {
    let ty::ProjectionPredicate { projection_ty, term } = pred;
    WherePredicate::EqPredicate {
        lhs: clean_projection(projection_ty, cx, None),
        rhs: clean_middle_term(term, cx),
    }
}

fn clean_projection<'tcx>(
    ty: ty::ProjectionTy<'tcx>,
    cx: &mut DocContext<'tcx>,
    def_id: Option<DefId>,
) -> Type {
    let lifted = ty.lift_to_tcx(cx.tcx).unwrap();
    let trait_ = clean_trait_ref_with_bindings(cx, lifted.trait_ref(cx.tcx), ThinVec::new());
    let self_type = clean_middle_ty(ty.self_ty(), cx, None);
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
        trait_,
    }))
}

fn compute_should_show_cast(self_def_id: Option<DefId>, trait_: &Path, self_type: &Type) -> bool {
    !trait_.segments.is_empty()
        && self_def_id
            .zip(Some(trait_.def_id()))
            .map_or(!self_type.is_self_type(), |(id, trait_)| id != trait_)
}

fn projection_to_path_segment<'tcx>(
    ty: ty::ProjectionTy<'tcx>,
    cx: &mut DocContext<'tcx>,
) -> PathSegment {
    let item = cx.tcx.associated_item(ty.item_def_id);
    let generics = cx.tcx.generics_of(ty.item_def_id);
    PathSegment {
        name: item.name,
        args: GenericArgs::AngleBracketed {
            args: substs_to_args(cx, &ty.substs[generics.parent_count..], false).into(),
            bindings: Default::default(),
        },
    }
}

fn clean_generic_param_def<'tcx>(
    def: &ty::GenericParamDef,
    cx: &mut DocContext<'tcx>,
) -> GenericParamDef {
    let (name, kind) = match def.kind {
        ty::GenericParamDefKind::Lifetime => {
            (def.name, GenericParamDefKind::Lifetime { outlives: vec![] })
        }
        ty::GenericParamDefKind::Type { has_default, synthetic, .. } => {
            let default = if has_default {
                Some(clean_middle_ty(cx.tcx.type_of(def.def_id), cx, Some(def.def_id)))
            } else {
                None
            };
            (
                def.name,
                GenericParamDefKind::Type {
                    did: def.def_id,
                    bounds: vec![], // These are filled in from the where-clauses.
                    default: default.map(Box::new),
                    synthetic,
                },
            )
        }
        ty::GenericParamDefKind::Const { has_default } => (
            def.name,
            GenericParamDefKind::Const {
                did: def.def_id,
                ty: Box::new(clean_middle_ty(cx.tcx.type_of(def.def_id), cx, Some(def.def_id))),
                default: match has_default {
                    true => Some(Box::new(cx.tcx.const_param_default(def.def_id).to_string())),
                    false => None,
                },
            },
        ),
    };

    GenericParamDef { name, kind }
}

fn clean_generic_param<'tcx>(
    cx: &mut DocContext<'tcx>,
    generics: Option<&hir::Generics<'tcx>>,
    param: &hir::GenericParam<'tcx>,
) -> GenericParamDef {
    let did = cx.tcx.hir().local_def_id(param.hir_id);
    let (name, kind) = match param.kind {
        hir::GenericParamKind::Lifetime { .. } => {
            let outlives = if let Some(generics) = generics {
                generics
                    .outlives_for_param(did)
                    .filter(|bp| !bp.in_where_clause)
                    .flat_map(|bp| bp.bounds)
                    .map(|bound| match bound {
                        hir::GenericBound::Outlives(lt) => clean_lifetime(lt, cx),
                        _ => panic!(),
                    })
                    .collect()
            } else {
                Vec::new()
            };
            (param.name.ident().name, GenericParamDefKind::Lifetime { outlives })
        }
        hir::GenericParamKind::Type { ref default, synthetic } => {
            let bounds = if let Some(generics) = generics {
                generics
                    .bounds_for_param(did)
                    .filter(|bp| bp.origin != PredicateOrigin::WhereClause)
                    .flat_map(|bp| bp.bounds)
                    .filter_map(|x| clean_generic_bound(x, cx))
                    .collect()
            } else {
                Vec::new()
            };
            (
                param.name.ident().name,
                GenericParamDefKind::Type {
                    did: did.to_def_id(),
                    bounds,
                    default: default.map(|t| clean_ty(t, cx)).map(Box::new),
                    synthetic,
                },
            )
        }
        hir::GenericParamKind::Const { ty, default } => (
            param.name.ident().name,
            GenericParamDefKind::Const {
                did: did.to_def_id(),
                ty: Box::new(clean_ty(ty, cx)),
                default: default.map(|ct| {
                    let def_id = cx.tcx.hir().local_def_id(ct.hir_id);
                    Box::new(ty::Const::from_anon_const(cx.tcx, def_id).to_string())
                }),
            },
        ),
    };

    GenericParamDef { name, kind }
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
    matches!(param.kind, hir::GenericParamKind::Lifetime { kind: hir::LifetimeParamKind::Elided })
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
                GenericParamDefKind::Type { did, ref bounds, .. } => {
                    cx.impl_trait_bounds.insert(did.into(), bounds.clone());
                }
                GenericParamDefKind::Const { .. } => unreachable!(),
            }
            param
        })
        .collect::<Vec<_>>();

    let mut params = Vec::with_capacity(gens.params.len());
    for p in gens.params.iter().filter(|p| !is_impl_trait(p) && !is_elided_lifetime(p)) {
        let p = clean_generic_param(cx, Some(gens), p);
        params.push(p);
    }
    params.extend(impl_trait_params);

    let mut generics = Generics {
        params,
        where_predicates: gens
            .predicates
            .iter()
            .filter_map(|x| clean_where_predicate(x, cx))
            .collect(),
    };

    // Some duplicates are generated for ?Sized bounds between type params and where
    // predicates. The point in here is to move the bounds definitions from type params
    // to where predicates when such cases occur.
    for where_pred in &mut generics.where_predicates {
        match *where_pred {
            WherePredicate::BoundPredicate { ty: Generic(ref name), ref mut bounds, .. } => {
                if bounds.is_empty() {
                    for param in &mut generics.params {
                        match param.kind {
                            GenericParamDefKind::Lifetime { .. } => {}
                            GenericParamDefKind::Type { bounds: ref mut ty_bounds, .. } => {
                                if &param.name == name {
                                    mem::swap(bounds, ty_bounds);
                                    break;
                                }
                            }
                            GenericParamDefKind::Const { .. } => {}
                        }
                    }
                }
            }
            _ => continue,
        }
    }
    generics
}

fn clean_ty_generics<'tcx>(
    cx: &mut DocContext<'tcx>,
    gens: &ty::Generics,
    preds: ty::GenericPredicates<'tcx>,
) -> Generics {
    // Don't populate `cx.impl_trait_bounds` before `clean`ning `where` clauses,
    // since `Clean for ty::Predicate` would consume them.
    let mut impl_trait = BTreeMap::<ImplTraitParam, Vec<GenericBound>>::default();

    // Bounds in the type_params and lifetimes fields are repeated in the
    // predicates field (see rustc_typeck::collect::ty_generics), so remove
    // them.
    let stripped_params = gens
        .params
        .iter()
        .filter_map(|param| match param.kind {
            ty::GenericParamDefKind::Lifetime if param.name == kw::UnderscoreLifetime => None,
            ty::GenericParamDefKind::Lifetime => Some(clean_generic_param_def(param, cx)),
            ty::GenericParamDefKind::Type { synthetic, .. } => {
                if param.name == kw::SelfUpper {
                    assert_eq!(param.index, 0);
                    return None;
                }
                if synthetic {
                    impl_trait.insert(param.index.into(), vec![]);
                    return None;
                }
                Some(clean_generic_param_def(param, cx))
            }
            ty::GenericParamDefKind::Const { .. } => Some(clean_generic_param_def(param, cx)),
        })
        .collect::<Vec<GenericParamDef>>();

    // param index -> [(DefId of trait, associated type name and generics, type)]
    let mut impl_trait_proj = FxHashMap::<u32, Vec<(DefId, PathSegment, Ty<'_>)>>::default();

    let where_predicates = preds
        .predicates
        .iter()
        .flat_map(|(p, _)| {
            let mut projection = None;
            let param_idx = (|| {
                let bound_p = p.kind();
                match bound_p.skip_binder() {
                    ty::PredicateKind::Trait(pred) => {
                        if let ty::Param(param) = pred.self_ty().kind() {
                            return Some(param.index);
                        }
                    }
                    ty::PredicateKind::TypeOutlives(ty::OutlivesPredicate(ty, _reg)) => {
                        if let ty::Param(param) = ty.kind() {
                            return Some(param.index);
                        }
                    }
                    ty::PredicateKind::Projection(p) => {
                        if let ty::Param(param) = p.projection_ty.self_ty().kind() {
                            projection = Some(bound_p.rebind(p));
                            return Some(param.index);
                        }
                    }
                    _ => (),
                }

                None
            })();

            if let Some(param_idx) = param_idx {
                if let Some(b) = impl_trait.get_mut(&param_idx.into()) {
                    let p: WherePredicate = clean_predicate(*p, cx)?;

                    b.extend(
                        p.get_bounds()
                            .into_iter()
                            .flatten()
                            .cloned()
                            .filter(|b| !b.is_sized_bound(cx)),
                    );

                    let proj = projection.map(|p| {
                        (
                            clean_projection(p.skip_binder().projection_ty, cx, None),
                            p.skip_binder().term,
                        )
                    });
                    if let Some(((_, trait_did, name), rhs)) = proj
                        .as_ref()
                        .and_then(|(lhs, rhs): &(Type, _)| Some((lhs.projection()?, rhs)))
                    {
                        // FIXME(...): Remove this unwrap()
                        impl_trait_proj.entry(param_idx).or_default().push((
                            trait_did,
                            name,
                            rhs.ty().unwrap(),
                        ));
                    }

                    return None;
                }
            }

            Some(p)
        })
        .collect::<Vec<_>>();

    for (param, mut bounds) in impl_trait {
        // Move trait bounds to the front.
        bounds.sort_by_key(|b| !matches!(b, GenericBound::TraitBound(..)));

        if let crate::core::ImplTraitParam::ParamIndex(idx) = param {
            if let Some(proj) = impl_trait_proj.remove(&idx) {
                for (trait_did, name, rhs) in proj {
                    let rhs = clean_middle_ty(rhs, cx, None);
                    simplify::merge_bounds(cx, &mut bounds, trait_did, name, &Term::Type(rhs));
                }
            }
        } else {
            unreachable!();
        }

        cx.impl_trait_bounds.insert(param, bounds);
    }

    // Now that `cx.impl_trait_bounds` is populated, we can process
    // remaining predicates which could contain `impl Trait`.
    let mut where_predicates =
        where_predicates.into_iter().flat_map(|p| clean_predicate(*p, cx)).collect::<Vec<_>>();

    // Type parameters have a Sized bound by default unless removed with
    // ?Sized. Scan through the predicates and mark any type parameter with
    // a Sized bound, removing the bounds as we find them.
    //
    // Note that associated types also have a sized bound by default, but we
    // don't actually know the set of associated types right here so that's
    // handled in cleaning associated types
    let mut sized_params = FxHashSet::default();
    where_predicates.retain(|pred| match *pred {
        WherePredicate::BoundPredicate { ty: Generic(ref g), ref bounds, .. } => {
            if bounds.iter().any(|b| b.is_sized_bound(cx)) {
                sized_params.insert(*g);
                false
            } else {
                true
            }
        }
        _ => true,
    });

    // Run through the type parameters again and insert a ?Sized
    // unbound for any we didn't find to be Sized.
    for tp in &stripped_params {
        if matches!(tp.kind, types::GenericParamDefKind::Type { .. })
            && !sized_params.contains(&tp.name)
        {
            where_predicates.push(WherePredicate::BoundPredicate {
                ty: Type::Generic(tp.name),
                bounds: vec![GenericBound::maybe_sized(cx)],
                bound_params: Vec::new(),
            })
        }
    }

    // It would be nice to collect all of the bounds on a type and recombine
    // them if possible, to avoid e.g., `where T: Foo, T: Bar, T: Sized, T: 'a`
    // and instead see `where T: Foo + Bar + Sized + 'a`

    Generics {
        params: stripped_params,
        where_predicates: simplify::where_clauses(cx, where_predicates),
    }
}

fn clean_fn_or_proc_macro<'tcx>(
    item: &hir::Item<'tcx>,
    sig: &hir::FnSig<'tcx>,
    generics: &hir::Generics<'tcx>,
    body_id: hir::BodyId,
    name: &mut Symbol,
    cx: &mut DocContext<'tcx>,
) -> ItemKind {
    let attrs = cx.tcx.hir().attrs(item.hir_id());
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
        Some(kind) => {
            if kind == MacroKind::Derive {
                *name = attrs
                    .lists(sym::proc_macro_derive)
                    .find_map(|mi| mi.ident())
                    .expect("proc-macro derives require a name")
                    .name;
            }

            let mut helpers = Vec::new();
            for mi in attrs.lists(sym::proc_macro_derive) {
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
        None => {
            let mut func = clean_function(cx, sig, generics, body_id);
            clean_fn_decl_legacy_const_generics(&mut func, attrs);
            FunctionItem(func)
        }
    }
}

/// This is needed to make it more "readable" when documenting functions using
/// `rustc_legacy_const_generics`. More information in
/// <https://github.com/rust-lang/rust/issues/83167>.
fn clean_fn_decl_legacy_const_generics(func: &mut Function, attrs: &[ast::Attribute]) {
    for meta_item_list in attrs
        .iter()
        .filter(|a| a.has_name(sym::rustc_legacy_const_generics))
        .filter_map(|a| a.meta_item_list())
    {
        for (pos, literal) in meta_item_list.iter().filter_map(|meta| meta.literal()).enumerate() {
            match literal.kind {
                ast::LitKind::Int(a, _) => {
                    let gen = func.generics.params.remove(0);
                    if let GenericParamDef { name, kind: GenericParamDefKind::Const { ty, .. } } =
                        gen
                    {
                        func.decl
                            .inputs
                            .values
                            .insert(a as _, Argument { name, type_: *ty, is_const: true });
                    } else {
                        panic!("unexpected non const in position {pos}");
                    }
                }
                _ => panic!("invalid arg index"),
            }
        }
    }
}

fn clean_function<'tcx>(
    cx: &mut DocContext<'tcx>,
    sig: &hir::FnSig<'tcx>,
    generics: &hir::Generics<'tcx>,
    body_id: hir::BodyId,
) -> Box<Function> {
    let (generics, decl) = enter_impl_trait(cx, |cx| {
        // NOTE: generics must be cleaned before args
        let generics = clean_generics(generics, cx);
        let args = clean_args_from_types_and_body_id(cx, sig.decl.inputs, body_id);
        let mut decl = clean_fn_decl_with_args(cx, sig.decl, args);
        if sig.header.is_async() {
            decl.output = decl.sugared_async_return_type();
        }
        (generics, decl)
    });
    Box::new(Function { decl, generics })
}

fn clean_args_from_types_and_names<'tcx>(
    cx: &mut DocContext<'tcx>,
    types: &[hir::Ty<'tcx>],
    names: &[Ident],
) -> Arguments {
    Arguments {
        values: types
            .iter()
            .enumerate()
            .map(|(i, ty)| {
                let mut name = names.get(i).map_or(kw::Empty, |ident| ident.name);
                if name.is_empty() {
                    name = kw::Underscore;
                }
                Argument { name, type_: clean_ty(ty, cx), is_const: false }
            })
            .collect(),
    }
}

fn clean_args_from_types_and_body_id<'tcx>(
    cx: &mut DocContext<'tcx>,
    types: &[hir::Ty<'tcx>],
    body_id: hir::BodyId,
) -> Arguments {
    let body = cx.tcx.hir().body(body_id);

    Arguments {
        values: types
            .iter()
            .enumerate()
            .map(|(i, ty)| Argument {
                name: name_from_pat(body.params[i].pat),
                type_: clean_ty(ty, cx),
                is_const: false,
            })
            .collect(),
    }
}

fn clean_fn_decl_with_args<'tcx>(
    cx: &mut DocContext<'tcx>,
    decl: &hir::FnDecl<'tcx>,
    args: Arguments,
) -> FnDecl {
    let output = match decl.output {
        hir::FnRetTy::Return(typ) => Return(clean_ty(typ, cx)),
        hir::FnRetTy::DefaultReturn(..) => DefaultReturn,
    };
    FnDecl { inputs: args, output, c_variadic: decl.c_variadic }
}

fn clean_fn_decl_from_did_and_sig<'tcx>(
    cx: &mut DocContext<'tcx>,
    did: Option<DefId>,
    sig: ty::PolyFnSig<'tcx>,
) -> FnDecl {
    let mut names = did.map_or(&[] as &[_], |did| cx.tcx.fn_arg_names(did)).iter();

    // We assume all empty tuples are default return type. This theoretically can discard `-> ()`,
    // but shouldn't change any code meaning.
    let output = match clean_middle_ty(sig.skip_binder().output(), cx, None) {
        Type::Tuple(inner) if inner.is_empty() => DefaultReturn,
        ty => Return(ty),
    };

    FnDecl {
        output,
        c_variadic: sig.skip_binder().c_variadic,
        inputs: Arguments {
            values: sig
                .skip_binder()
                .inputs()
                .iter()
                .map(|t| Argument {
                    type_: clean_middle_ty(*t, cx, None),
                    name: names.next().map_or(kw::Empty, |i| i.name),
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
    let local_did = trait_item.def_id.to_def_id();
    cx.with_param_env(local_did, |cx| {
        let inner = match trait_item.kind {
            hir::TraitItemKind::Const(ty, Some(default)) => AssocConstItem(
                clean_ty(ty, cx),
                ConstantKind::Local { def_id: local_did, body: default },
            ),
            hir::TraitItemKind::Const(ty, None) => TyAssocConstItem(clean_ty(ty, cx)),
            hir::TraitItemKind::Fn(ref sig, hir::TraitFn::Provided(body)) => {
                let m = clean_function(cx, sig, trait_item.generics, body);
                MethodItem(m, None)
            }
            hir::TraitItemKind::Fn(ref sig, hir::TraitFn::Required(names)) => {
                let (generics, decl) = enter_impl_trait(cx, |cx| {
                    // NOTE: generics must be cleaned before args
                    let generics = clean_generics(trait_item.generics, cx);
                    let args = clean_args_from_types_and_names(cx, sig.decl.inputs, names);
                    let decl = clean_fn_decl_with_args(cx, sig.decl, args);
                    (generics, decl)
                });
                TyMethodItem(Box::new(Function { decl, generics }))
            }
            hir::TraitItemKind::Type(bounds, Some(default)) => {
                let generics = enter_impl_trait(cx, |cx| clean_generics(trait_item.generics, cx));
                let bounds = bounds.iter().filter_map(|x| clean_generic_bound(x, cx)).collect();
                let item_type = clean_middle_ty(hir_ty_to_ty(cx.tcx, default), cx, None);
                AssocTypeItem(
                    Box::new(Typedef {
                        type_: clean_ty(default, cx),
                        generics,
                        item_type: Some(item_type),
                    }),
                    bounds,
                )
            }
            hir::TraitItemKind::Type(bounds, None) => {
                let generics = enter_impl_trait(cx, |cx| clean_generics(trait_item.generics, cx));
                let bounds = bounds.iter().filter_map(|x| clean_generic_bound(x, cx)).collect();
                TyAssocTypeItem(Box::new(generics), bounds)
            }
        };
        let what_rustc_thinks =
            Item::from_def_id_and_parts(local_did, Some(trait_item.ident.name), inner, cx);
        // Trait items always inherit the trait's visibility -- we don't want to show `pub`.
        Item { visibility: Inherited, ..what_rustc_thinks }
    })
}

pub(crate) fn clean_impl_item<'tcx>(
    impl_: &hir::ImplItem<'tcx>,
    cx: &mut DocContext<'tcx>,
) -> Item {
    let local_did = impl_.def_id.to_def_id();
    cx.with_param_env(local_did, |cx| {
        let inner = match impl_.kind {
            hir::ImplItemKind::Const(ty, expr) => {
                let default = ConstantKind::Local { def_id: local_did, body: expr };
                AssocConstItem(clean_ty(ty, cx), default)
            }
            hir::ImplItemKind::Fn(ref sig, body) => {
                let m = clean_function(cx, sig, impl_.generics, body);
                let defaultness = cx.tcx.impl_defaultness(impl_.def_id);
                MethodItem(m, Some(defaultness))
            }
            hir::ImplItemKind::TyAlias(hir_ty) => {
                let type_ = clean_ty(hir_ty, cx);
                let generics = clean_generics(impl_.generics, cx);
                let item_type = clean_middle_ty(hir_ty_to_ty(cx.tcx, hir_ty), cx, None);
                AssocTypeItem(
                    Box::new(Typedef { type_, generics, item_type: Some(item_type) }),
                    Vec::new(),
                )
            }
        };

        let mut what_rustc_thinks =
            Item::from_def_id_and_parts(local_did, Some(impl_.ident.name), inner, cx);

        let impl_ref = cx.tcx.impl_trait_ref(cx.tcx.local_parent(impl_.def_id));

        // Trait impl items always inherit the impl's visibility --
        // we don't want to show `pub`.
        if impl_ref.is_some() {
            what_rustc_thinks.visibility = Inherited;
        }

        what_rustc_thinks
    })
}

pub(crate) fn clean_middle_assoc_item<'tcx>(
    assoc_item: &ty::AssocItem,
    cx: &mut DocContext<'tcx>,
) -> Item {
    let tcx = cx.tcx;
    let kind = match assoc_item.kind {
        ty::AssocKind::Const => {
            let ty = clean_middle_ty(tcx.type_of(assoc_item.def_id), cx, Some(assoc_item.def_id));

            let provided = match assoc_item.container {
                ty::ImplContainer => true,
                ty::TraitContainer => tcx.impl_defaultness(assoc_item.def_id).has_value(),
            };
            if provided {
                AssocConstItem(ty, ConstantKind::Extern { def_id: assoc_item.def_id })
            } else {
                TyAssocConstItem(ty)
            }
        }
        ty::AssocKind::Fn => {
            let generics = clean_ty_generics(
                cx,
                tcx.generics_of(assoc_item.def_id),
                tcx.explicit_predicates_of(assoc_item.def_id),
            );
            let sig = tcx.fn_sig(assoc_item.def_id);
            let mut decl = clean_fn_decl_from_did_and_sig(cx, Some(assoc_item.def_id), sig);

            if assoc_item.fn_has_self_parameter {
                let self_ty = match assoc_item.container {
                    ty::ImplContainer => tcx.type_of(assoc_item.container_id(tcx)),
                    ty::TraitContainer => tcx.types.self_param,
                };
                let self_arg_ty = sig.input(0).skip_binder();
                if self_arg_ty == self_ty {
                    decl.inputs.values[0].type_ = Generic(kw::SelfUpper);
                } else if let ty::Ref(_, ty, _) = *self_arg_ty.kind() {
                    if ty == self_ty {
                        match decl.inputs.values[0].type_ {
                            BorrowedRef { ref mut type_, .. } => **type_ = Generic(kw::SelfUpper),
                            _ => unreachable!(),
                        }
                    }
                }
            }

            let provided = match assoc_item.container {
                ty::ImplContainer => true,
                ty::TraitContainer => assoc_item.defaultness(tcx).has_value(),
            };
            if provided {
                let defaultness = match assoc_item.container {
                    ty::ImplContainer => Some(assoc_item.defaultness(tcx)),
                    ty::TraitContainer => None,
                };
                MethodItem(Box::new(Function { generics, decl }), defaultness)
            } else {
                TyMethodItem(Box::new(Function { generics, decl }))
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
                    (GenericParamDefKind::Const { .. }, GenericArg::Const(c)) => match &c.kind {
                        ConstantKind::TyConst { expr } => expr == param.name.as_str(),
                        _ => false,
                    },
                    _ => false,
                }
            }

            if let ty::TraitContainer = assoc_item.container {
                let bounds = tcx.explicit_item_bounds(assoc_item.def_id);
                let predicates = ty::GenericPredicates { parent: None, predicates: bounds };
                let mut generics =
                    clean_ty_generics(cx, tcx.generics_of(assoc_item.def_id), predicates);
                // Filter out the bounds that are (likely?) directly attached to the associated type,
                // as opposed to being located in the where clause.
                let mut bounds = generics
                    .where_predicates
                    .drain_filter(|pred| match *pred {
                        WherePredicate::BoundPredicate {
                            ty: QPath(box QPathData { ref assoc, ref self_type, ref trait_, .. }),
                            ..
                        } => {
                            if assoc.name != my_name {
                                return false;
                            }
                            if trait_.def_id() != assoc_item.container_id(tcx) {
                                return false;
                            }
                            match *self_type {
                                Generic(ref s) if *s == kw::SelfUpper => {}
                                _ => return false,
                            }
                            match &assoc.args {
                                GenericArgs::AngleBracketed { args, bindings } => {
                                    if !bindings.is_empty()
                                        || generics
                                            .params
                                            .iter()
                                            .zip(args.iter())
                                            .any(|(param, arg)| !param_eq_arg(param, arg))
                                    {
                                        return false;
                                    }
                                }
                                GenericArgs::Parenthesized { .. } => {
                                    // The only time this happens is if we're inside the rustdoc for Fn(),
                                    // which only has one associated type, which is not a GAT, so whatever.
                                }
                            }
                            true
                        }
                        _ => false,
                    })
                    .flat_map(|pred| {
                        if let WherePredicate::BoundPredicate { bounds, .. } = pred {
                            bounds
                        } else {
                            unreachable!()
                        }
                    })
                    .collect::<Vec<_>>();
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

                if tcx.impl_defaultness(assoc_item.def_id).has_value() {
                    AssocTypeItem(
                        Box::new(Typedef {
                            type_: clean_middle_ty(
                                tcx.type_of(assoc_item.def_id),
                                cx,
                                Some(assoc_item.def_id),
                            ),
                            generics,
                            // FIXME: should we obtain the Type from HIR and pass it on here?
                            item_type: None,
                        }),
                        bounds,
                    )
                } else {
                    TyAssocTypeItem(Box::new(generics), bounds)
                }
            } else {
                // FIXME: when could this happen? Associated items in inherent impls?
                AssocTypeItem(
                    Box::new(Typedef {
                        type_: clean_middle_ty(
                            tcx.type_of(assoc_item.def_id),
                            cx,
                            Some(assoc_item.def_id),
                        ),
                        generics: Generics { params: Vec::new(), where_predicates: Vec::new() },
                        item_type: None,
                    }),
                    Vec::new(),
                )
            }
        }
    };

    let mut what_rustc_thinks =
        Item::from_def_id_and_parts(assoc_item.def_id, Some(assoc_item.name), kind, cx);

    let impl_ref = tcx.impl_trait_ref(tcx.parent(assoc_item.def_id));

    // Trait impl items always inherit the impl's visibility --
    // we don't want to show `pub`.
    if impl_ref.is_some() {
        what_rustc_thinks.visibility = Visibility::Inherited;
    }

    what_rustc_thinks
}

fn clean_qpath<'tcx>(hir_ty: &hir::Ty<'tcx>, cx: &mut DocContext<'tcx>) -> Type {
    let hir::Ty { hir_id: _, span, ref kind } = *hir_ty;
    let hir::TyKind::Path(qpath) = kind else { unreachable!() };

    match qpath {
        hir::QPath::Resolved(None, path) => {
            if let Res::Def(DefKind::TyParam, did) = path.res {
                if let Some(new_ty) = cx.substs.get(&did).and_then(|p| p.as_ty()).cloned() {
                    return new_ty;
                }
                if let Some(bounds) = cx.impl_trait_bounds.remove(&did.into()) {
                    return ImplTrait(bounds);
                }
            }

            if let Some(expanded) = maybe_expand_private_type_alias(cx, path) {
                expanded
            } else {
                let path = clean_path(path, cx);
                resolve_type(cx, path)
            }
        }
        hir::QPath::Resolved(Some(qself), p) => {
            // Try to normalize `<X as Y>::T` to a type
            let ty = hir_ty_to_ty(cx.tcx, hir_ty);
            if let Some(normalized_value) = normalize(cx, ty) {
                return clean_middle_ty(normalized_value, cx, None);
            }

            let trait_segments = &p.segments[..p.segments.len() - 1];
            let trait_def = cx.tcx.associated_item(p.res.def_id()).container_id(cx.tcx);
            let trait_ = self::Path {
                res: Res::Def(DefKind::Trait, trait_def),
                segments: trait_segments.iter().map(|x| clean_path_segment(x, cx)).collect(),
            };
            register_res(cx, trait_.res);
            let self_def_id = DefId::local(qself.hir_id.owner.local_def_index);
            let self_type = clean_ty(qself, cx);
            let should_show_cast = compute_should_show_cast(Some(self_def_id), &trait_, &self_type);
            Type::QPath(Box::new(QPathData {
                assoc: clean_path_segment(p.segments.last().expect("segments were empty"), cx),
                should_show_cast,
                self_type,
                trait_,
            }))
        }
        hir::QPath::TypeRelative(qself, segment) => {
            let ty = hir_ty_to_ty(cx.tcx, hir_ty);
            let res = match ty.kind() {
                ty::Projection(proj) => Res::Def(DefKind::Trait, proj.trait_ref(cx.tcx).def_id),
                // Rustdoc handles `ty::Error`s by turning them into `Type::Infer`s.
                ty::Error(_) => return Type::Infer,
                _ => bug!("clean: expected associated type, found `{:?}`", ty),
            };
            let trait_ = clean_path(&hir::Path { span, res, segments: &[] }, cx);
            register_res(cx, trait_.res);
            let self_def_id = res.opt_def_id();
            let self_type = clean_ty(qself, cx);
            let should_show_cast = compute_should_show_cast(self_def_id, &trait_, &self_type);
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
    let alias = if !cx.cache.access_levels.is_exported(def_id.to_def_id()) {
        &cx.tcx.hir().expect_item(def_id).kind
    } else {
        return None;
    };
    let hir::ItemKind::TyAlias(ty, generics) = alias else { return None };

    let provided_params = &path.segments.last().expect("segments were empty");
    let mut substs = FxHashMap::default();
    let generic_args = provided_params.args();

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
                    let lt_def_id = cx.tcx.hir().local_def_id(param.hir_id);
                    let cleaned =
                        if !lt.is_elided() { clean_lifetime(lt, cx) } else { Lifetime::elided() };
                    substs.insert(lt_def_id.to_def_id(), SubstParam::Lifetime(cleaned));
                }
                indices.lifetimes += 1;
            }
            hir::GenericParamKind::Type { ref default, .. } => {
                let ty_param_def_id = cx.tcx.hir().local_def_id(param.hir_id);
                let mut j = 0;
                let type_ = generic_args.args.iter().find_map(|arg| match arg {
                    hir::GenericArg::Type(ty) => {
                        if indices.types == j {
                            return Some(ty);
                        }
                        j += 1;
                        None
                    }
                    _ => None,
                });
                if let Some(ty) = type_ {
                    substs.insert(ty_param_def_id.to_def_id(), SubstParam::Type(clean_ty(ty, cx)));
                } else if let Some(default) = *default {
                    substs.insert(
                        ty_param_def_id.to_def_id(),
                        SubstParam::Type(clean_ty(default, cx)),
                    );
                }
                indices.types += 1;
            }
            hir::GenericParamKind::Const { .. } => {
                let const_param_def_id = cx.tcx.hir().local_def_id(param.hir_id);
                let mut j = 0;
                let const_ = generic_args.args.iter().find_map(|arg| match arg {
                    hir::GenericArg::Const(ct) => {
                        if indices.consts == j {
                            return Some(ct);
                        }
                        j += 1;
                        None
                    }
                    _ => None,
                });
                if let Some(ct) = const_ {
                    substs.insert(
                        const_param_def_id.to_def_id(),
                        SubstParam::Constant(clean_const(ct, cx)),
                    );
                }
                // FIXME(const_generics_defaults)
                indices.consts += 1;
            }
        }
    }

    Some(cx.enter_alias(substs, |cx| clean_ty(ty, cx)))
}

pub(crate) fn clean_ty<'tcx>(ty: &hir::Ty<'tcx>, cx: &mut DocContext<'tcx>) -> Type {
    use rustc_hir::*;

    match ty.kind {
        TyKind::Never => Primitive(PrimitiveType::Never),
        TyKind::Ptr(ref m) => RawPointer(m.mutbl, Box::new(clean_ty(m.ty, cx))),
        TyKind::Rptr(ref l, ref m) => {
            // There are two times a `Fresh` lifetime can be created:
            // 1. For `&'_ x`, written by the user. This corresponds to `lower_lifetime` in `rustc_ast_lowering`.
            // 2. For `&x` as a parameter to an `async fn`. This corresponds to `elided_ref_lifetime in `rustc_ast_lowering`.
            //    See #59286 for more information.
            // Ideally we would only hide the `'_` for case 2., but I don't know a way to distinguish it.
            // Turning `fn f(&'_ self)` into `fn f(&self)` isn't the worst thing in the world, though;
            // there's no case where it could cause the function to fail to compile.
            let elided =
                l.is_elided() || matches!(l.name, LifetimeName::Param(_, ParamName::Fresh));
            let lifetime = if elided { None } else { Some(clean_lifetime(*l, cx)) };
            BorrowedRef { lifetime, mutability: m.mutbl, type_: Box::new(clean_ty(m.ty, cx)) }
        }
        TyKind::Slice(ty) => Slice(Box::new(clean_ty(ty, cx))),
        TyKind::Array(ty, ref length) => {
            let length = match length {
                hir::ArrayLen::Infer(_, _) => "_".to_string(),
                hir::ArrayLen::Body(anon_const) => {
                    let def_id = cx.tcx.hir().local_def_id(anon_const.hir_id);
                    // NOTE(min_const_generics): We can't use `const_eval_poly` for constants
                    // as we currently do not supply the parent generics to anonymous constants
                    // but do allow `ConstKind::Param`.
                    //
                    // `const_eval_poly` tries to to first substitute generic parameters which
                    // results in an ICE while manually constructing the constant and using `eval`
                    // does nothing for `ConstKind::Param`.
                    let ct = ty::Const::from_anon_const(cx.tcx, def_id);
                    let param_env = cx.tcx.param_env(def_id);
                    print_const(cx, ct.eval(cx.tcx, param_env))
                }
            };

            Array(Box::new(clean_ty(ty, cx)), length)
        }
        TyKind::Tup(tys) => Tuple(tys.iter().map(|ty| clean_ty(ty, cx)).collect()),
        TyKind::OpaqueDef(item_id, _) => {
            let item = cx.tcx.hir().item(item_id);
            if let hir::ItemKind::OpaqueTy(ref ty) = item.kind {
                ImplTrait(ty.bounds.iter().filter_map(|x| clean_generic_bound(x, cx)).collect())
            } else {
                unreachable!()
            }
        }
        TyKind::Path(_) => clean_qpath(ty, cx),
        TyKind::TraitObject(bounds, ref lifetime, _) => {
            let bounds = bounds.iter().map(|bound| clean_poly_trait_ref(bound, cx)).collect();
            let lifetime =
                if !lifetime.is_elided() { Some(clean_lifetime(*lifetime, cx)) } else { None };
            DynTrait(bounds, lifetime)
        }
        TyKind::BareFn(barefn) => BareFunction(Box::new(clean_bare_fn_ty(barefn, cx))),
        // Rustdoc handles `TyKind::Err`s by turning them into `Type::Infer`s.
        TyKind::Infer | TyKind::Err => Infer,
        TyKind::Typeof(..) => panic!("unimplemented type {:?}", ty.kind),
    }
}

/// Returns `None` if the type could not be normalized
fn normalize<'tcx>(cx: &mut DocContext<'tcx>, ty: Ty<'_>) -> Option<Ty<'tcx>> {
    // HACK: low-churn fix for #79459 while we wait for a trait normalization fix
    if !cx.tcx.sess.opts.unstable_opts.normalize_docs {
        return None;
    }

    use crate::rustc_trait_selection::infer::TyCtxtInferExt;
    use crate::rustc_trait_selection::traits::query::normalize::AtExt;
    use rustc_middle::traits::ObligationCause;

    // Try to normalize `<X as Y>::T` to a type
    let lifted = ty.lift_to_tcx(cx.tcx).unwrap();
    let normalized = cx.tcx.infer_ctxt().enter(|infcx| {
        infcx
            .at(&ObligationCause::dummy(), cx.param_env)
            .normalize(lifted)
            .map(|resolved| infcx.resolve_vars_if_possible(resolved.value))
    });
    match normalized {
        Ok(normalized_value) => {
            debug!("normalized {:?} to {:?}", ty, normalized_value);
            Some(normalized_value)
        }
        Err(err) => {
            debug!("failed to normalize {:?}: {:?}", ty, err);
            None
        }
    }
}

pub(crate) fn clean_middle_ty<'tcx>(
    this: Ty<'tcx>,
    cx: &mut DocContext<'tcx>,
    def_id: Option<DefId>,
) -> Type {
    trace!("cleaning type: {:?}", this);
    let ty = normalize(cx, this).unwrap_or(this);
    match *ty.kind() {
        ty::Never => Primitive(PrimitiveType::Never),
        ty::Bool => Primitive(PrimitiveType::Bool),
        ty::Char => Primitive(PrimitiveType::Char),
        ty::Int(int_ty) => Primitive(int_ty.into()),
        ty::Uint(uint_ty) => Primitive(uint_ty.into()),
        ty::Float(float_ty) => Primitive(float_ty.into()),
        ty::Str => Primitive(PrimitiveType::Str),
        ty::Slice(ty) => Slice(Box::new(clean_middle_ty(ty, cx, None))),
        ty::Array(ty, n) => {
            let mut n = cx.tcx.lift(n).expect("array lift failed");
            n = n.eval(cx.tcx, ty::ParamEnv::reveal_all());
            let n = print_const(cx, n);
            Array(Box::new(clean_middle_ty(ty, cx, None)), n)
        }
        ty::RawPtr(mt) => RawPointer(mt.mutbl, Box::new(clean_middle_ty(mt.ty, cx, None))),
        ty::Ref(r, ty, mutbl) => BorrowedRef {
            lifetime: clean_middle_region(r),
            mutability: mutbl,
            type_: Box::new(clean_middle_ty(ty, cx, None)),
        },
        ty::FnDef(..) | ty::FnPtr(_) => {
            let ty = cx.tcx.lift(this).expect("FnPtr lift failed");
            let sig = ty.fn_sig(cx.tcx);
            let decl = clean_fn_decl_from_did_and_sig(cx, None, sig);
            BareFunction(Box::new(BareFunctionDecl {
                unsafety: sig.unsafety(),
                generic_params: Vec::new(),
                decl,
                abi: sig.abi(),
            }))
        }
        ty::Adt(def, substs) => {
            let did = def.did();
            let kind = match def.adt_kind() {
                AdtKind::Struct => ItemType::Struct,
                AdtKind::Union => ItemType::Union,
                AdtKind::Enum => ItemType::Enum,
            };
            inline::record_extern_fqn(cx, did, kind);
            let path = external_path(cx, did, false, ThinVec::new(), substs);
            Type::Path { path }
        }
        ty::Foreign(did) => {
            inline::record_extern_fqn(cx, did, ItemType::ForeignType);
            let path = external_path(cx, did, false, ThinVec::new(), InternalSubsts::empty());
            Type::Path { path }
        }
        ty::Dynamic(obj, ref reg) => {
            // HACK: pick the first `did` as the `did` of the trait object. Someone
            // might want to implement "native" support for marker-trait-only
            // trait objects.
            let mut dids = obj.auto_traits();
            let did = obj
                .principal_def_id()
                .or_else(|| dids.next())
                .unwrap_or_else(|| panic!("found trait object `{:?}` with no traits?", this));
            let substs = match obj.principal() {
                Some(principal) => principal.skip_binder().substs,
                // marker traits have no substs.
                _ => cx.tcx.intern_substs(&[]),
            };

            inline::record_extern_fqn(cx, did, ItemType::Trait);

            let lifetime = clean_middle_region(*reg);
            let mut bounds = dids
                .map(|did| {
                    let empty = cx.tcx.intern_substs(&[]);
                    let path = external_path(cx, did, false, ThinVec::new(), empty);
                    inline::record_extern_fqn(cx, did, ItemType::Trait);
                    PolyTrait { trait_: path, generic_params: Vec::new() }
                })
                .collect::<Vec<_>>();

            let bindings = obj
                .projection_bounds()
                .map(|pb| TypeBinding {
                    assoc: projection_to_path_segment(
                        pb.skip_binder()
                            .lift_to_tcx(cx.tcx)
                            .unwrap()
                            // HACK(compiler-errors): Doesn't actually matter what self
                            // type we put here, because we're only using the GAT's substs.
                            .with_self_ty(cx.tcx, cx.tcx.types.self_param)
                            .projection_ty,
                        cx,
                    ),
                    kind: TypeBindingKind::Equality {
                        term: clean_middle_term(pb.skip_binder().term, cx),
                    },
                })
                .collect();

            let path = external_path(cx, did, false, bindings, substs);
            bounds.insert(0, PolyTrait { trait_: path, generic_params: Vec::new() });

            DynTrait(bounds, lifetime)
        }
        ty::Tuple(t) => Tuple(t.iter().map(|t| clean_middle_ty(t, cx, None)).collect()),

        ty::Projection(ref data) => clean_projection(*data, cx, def_id),

        ty::Param(ref p) => {
            if let Some(bounds) = cx.impl_trait_bounds.remove(&p.index.into()) {
                ImplTrait(bounds)
            } else {
                Generic(p.name)
            }
        }

        ty::Opaque(def_id, substs) => {
            // Grab the "TraitA + TraitB" from `impl TraitA + TraitB`,
            // by looking up the bounds associated with the def_id.
            let substs = cx.tcx.lift(substs).expect("Opaque lift failed");
            let bounds = cx
                .tcx
                .explicit_item_bounds(def_id)
                .iter()
                .map(|(bound, _)| EarlyBinder(*bound).subst(cx.tcx, substs))
                .collect::<Vec<_>>();
            let mut regions = vec![];
            let mut has_sized = false;
            let mut bounds = bounds
                .iter()
                .filter_map(|bound| {
                    let bound_predicate = bound.kind();
                    let trait_ref = match bound_predicate.skip_binder() {
                        ty::PredicateKind::Trait(tr) => bound_predicate.rebind(tr.trait_ref),
                        ty::PredicateKind::TypeOutlives(ty::OutlivesPredicate(_ty, reg)) => {
                            if let Some(r) = clean_middle_region(reg) {
                                regions.push(GenericBound::Outlives(r));
                            }
                            return None;
                        }
                        _ => return None,
                    };

                    if let Some(sized) = cx.tcx.lang_items().sized_trait() {
                        if trait_ref.def_id() == sized {
                            has_sized = true;
                            return None;
                        }
                    }

                    let bindings: ThinVec<_> = bounds
                        .iter()
                        .filter_map(|bound| {
                            if let ty::PredicateKind::Projection(proj) = bound.kind().skip_binder()
                            {
                                if proj.projection_ty.trait_ref(cx.tcx) == trait_ref.skip_binder() {
                                    Some(TypeBinding {
                                        assoc: projection_to_path_segment(proj.projection_ty, cx),
                                        kind: TypeBindingKind::Equality {
                                            term: clean_middle_term(proj.term, cx),
                                        },
                                    })
                                } else {
                                    None
                                }
                            } else {
                                None
                            }
                        })
                        .collect();

                    Some(clean_poly_trait_ref_with_bindings(cx, trait_ref, bindings))
                })
                .collect::<Vec<_>>();
            bounds.extend(regions);
            if !has_sized && !bounds.is_empty() {
                bounds.insert(0, GenericBound::maybe_sized(cx));
            }
            ImplTrait(bounds)
        }

        ty::Closure(..) => panic!("Closure"),
        ty::Generator(..) => panic!("Generator"),
        ty::Bound(..) => panic!("Bound"),
        ty::Placeholder(..) => panic!("Placeholder"),
        ty::GeneratorWitness(..) => panic!("GeneratorWitness"),
        ty::Infer(..) => panic!("Infer"),
        ty::Error(_) => panic!("Error"),
    }
}

pub(crate) fn clean_field<'tcx>(field: &hir::FieldDef<'tcx>, cx: &mut DocContext<'tcx>) -> Item {
    let def_id = cx.tcx.hir().local_def_id(field.hir_id).to_def_id();
    clean_field_with_def_id(def_id, field.ident.name, clean_ty(field.ty, cx), cx)
}

pub(crate) fn clean_middle_field<'tcx>(field: &ty::FieldDef, cx: &mut DocContext<'tcx>) -> Item {
    clean_field_with_def_id(
        field.did,
        field.name,
        clean_middle_ty(cx.tcx.type_of(field.did), cx, Some(field.did)),
        cx,
    )
}

pub(crate) fn clean_field_with_def_id(
    def_id: DefId,
    name: Symbol,
    ty: Type,
    cx: &mut DocContext<'_>,
) -> Item {
    let what_rustc_thinks =
        Item::from_def_id_and_parts(def_id, Some(name), StructFieldItem(ty), cx);
    if is_field_vis_inherited(cx.tcx, def_id) {
        // Variant fields inherit their enum's visibility.
        Item { visibility: Visibility::Inherited, ..what_rustc_thinks }
    } else {
        what_rustc_thinks
    }
}

fn is_field_vis_inherited(tcx: TyCtxt<'_>, def_id: DefId) -> bool {
    let parent = tcx.parent(def_id);
    match tcx.def_kind(parent) {
        DefKind::Struct | DefKind::Union => false,
        DefKind::Variant => true,
        parent_kind => panic!("unexpected parent kind: {:?}", parent_kind),
    }
}

pub(crate) fn clean_visibility(vis: ty::Visibility<DefId>) -> Visibility {
    match vis {
        ty::Visibility::Public => Visibility::Public,
        ty::Visibility::Restricted(module) => Visibility::Restricted(module),
    }
}

pub(crate) fn clean_variant_def<'tcx>(variant: &ty::VariantDef, cx: &mut DocContext<'tcx>) -> Item {
    let kind = match variant.ctor_kind {
        CtorKind::Const => Variant::CLike(match variant.discr {
            ty::VariantDiscr::Explicit(def_id) => Some(Discriminant { expr: None, value: def_id }),
            ty::VariantDiscr::Relative(_) => None,
        }),
        CtorKind::Fn => Variant::Tuple(
            variant.fields.iter().map(|field| clean_middle_field(field, cx)).collect(),
        ),
        CtorKind::Fictive => Variant::Struct(VariantStruct {
            struct_type: CtorKind::Fictive,
            fields: variant.fields.iter().map(|field| clean_middle_field(field, cx)).collect(),
        }),
    };
    let what_rustc_thinks =
        Item::from_def_id_and_parts(variant.def_id, Some(variant.name), VariantItem(kind), cx);
    // don't show `pub` for variants, which always inherit visibility
    Item { visibility: Inherited, ..what_rustc_thinks }
}

fn clean_variant_data<'tcx>(
    variant: &hir::VariantData<'tcx>,
    disr_expr: &Option<hir::AnonConst>,
    cx: &mut DocContext<'tcx>,
) -> Variant {
    match variant {
        hir::VariantData::Struct(..) => Variant::Struct(VariantStruct {
            struct_type: CtorKind::from_hir(variant),
            fields: variant.fields().iter().map(|x| clean_field(x, cx)).collect(),
        }),
        hir::VariantData::Tuple(..) => {
            Variant::Tuple(variant.fields().iter().map(|x| clean_field(x, cx)).collect())
        }
        hir::VariantData::Unit(..) => Variant::CLike(disr_expr.map(|disr| Discriminant {
            expr: Some(disr.body),
            value: cx.tcx.hir().local_def_id(disr.hir_id).to_def_id(),
        })),
    }
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
    if generic_args.parenthesized {
        let output = clean_ty(generic_args.bindings[0].ty(), cx);
        let output = if output != Type::Tuple(Vec::new()) { Some(Box::new(output)) } else { None };
        let inputs =
            generic_args.inputs().iter().map(|x| clean_ty(x, cx)).collect::<Vec<_>>().into();
        GenericArgs::Parenthesized { inputs, output }
    } else {
        let args = generic_args
            .args
            .iter()
            .map(|arg| match arg {
                hir::GenericArg::Lifetime(lt) if !lt.is_elided() => {
                    GenericArg::Lifetime(clean_lifetime(*lt, cx))
                }
                hir::GenericArg::Lifetime(_) => GenericArg::Lifetime(Lifetime::elided()),
                hir::GenericArg::Type(ty) => GenericArg::Type(clean_ty(ty, cx)),
                hir::GenericArg::Const(ct) => GenericArg::Const(Box::new(clean_const(ct, cx))),
                hir::GenericArg::Infer(_inf) => GenericArg::Infer,
            })
            .collect::<Vec<_>>()
            .into();
        let bindings =
            generic_args.bindings.iter().map(|x| clean_type_binding(x, cx)).collect::<ThinVec<_>>();
        GenericArgs::AngleBracketed { args, bindings }
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
        let decl = clean_fn_decl_with_args(cx, bare_fn.decl, args);
        (generic_params, decl)
    });
    BareFunctionDecl { unsafety: bare_fn.unsafety, abi: bare_fn.abi, decl, generic_params }
}

fn clean_maybe_renamed_item<'tcx>(
    cx: &mut DocContext<'tcx>,
    item: &hir::Item<'tcx>,
    renamed: Option<Symbol>,
) -> Vec<Item> {
    use hir::ItemKind;

    let def_id = item.def_id.to_def_id();
    let mut name = renamed.unwrap_or_else(|| cx.tcx.hir().name(item.hir_id()));
    cx.with_param_env(def_id, |cx| {
        let kind = match item.kind {
            ItemKind::Static(ty, mutability, body_id) => {
                StaticItem(Static { type_: clean_ty(ty, cx), mutability, expr: Some(body_id) })
            }
            ItemKind::Const(ty, body_id) => ConstantItem(Constant {
                type_: clean_ty(ty, cx),
                kind: ConstantKind::Local { body: body_id, def_id },
            }),
            ItemKind::OpaqueTy(ref ty) => OpaqueTyItem(OpaqueTy {
                bounds: ty.bounds.iter().filter_map(|x| clean_generic_bound(x, cx)).collect(),
                generics: clean_generics(ty.generics, cx),
            }),
            ItemKind::TyAlias(hir_ty, generics) => {
                let rustdoc_ty = clean_ty(hir_ty, cx);
                let ty = clean_middle_ty(hir_ty_to_ty(cx.tcx, hir_ty), cx, None);
                TypedefItem(Box::new(Typedef {
                    type_: rustdoc_ty,
                    generics: clean_generics(generics, cx),
                    item_type: Some(ty),
                }))
            }
            ItemKind::Enum(ref def, generics) => EnumItem(Enum {
                variants: def.variants.iter().map(|v| clean_variant(v, cx)).collect(),
                generics: clean_generics(generics, cx),
            }),
            ItemKind::TraitAlias(generics, bounds) => TraitAliasItem(TraitAlias {
                generics: clean_generics(generics, cx),
                bounds: bounds.iter().filter_map(|x| clean_generic_bound(x, cx)).collect(),
            }),
            ItemKind::Union(ref variant_data, generics) => UnionItem(Union {
                generics: clean_generics(generics, cx),
                fields: variant_data.fields().iter().map(|x| clean_field(x, cx)).collect(),
            }),
            ItemKind::Struct(ref variant_data, generics) => StructItem(Struct {
                struct_type: CtorKind::from_hir(variant_data),
                generics: clean_generics(generics, cx),
                fields: variant_data.fields().iter().map(|x| clean_field(x, cx)).collect(),
            }),
            ItemKind::Impl(impl_) => return clean_impl(impl_, item.hir_id(), cx),
            // proc macros can have a name set by attributes
            ItemKind::Fn(ref sig, generics, body_id) => {
                clean_fn_or_proc_macro(item, sig, generics, body_id, &mut name, cx)
            }
            ItemKind::Macro(ref macro_def, _) => {
                let ty_vis = clean_visibility(cx.tcx.visibility(def_id));
                MacroItem(Macro {
                    source: display_macro_source(cx, name, macro_def, def_id, ty_vis),
                })
            }
            ItemKind::Trait(_, _, generics, bounds, item_ids) => {
                let items = item_ids
                    .iter()
                    .map(|ti| clean_trait_item(cx.tcx.hir().trait_item(ti.id), cx))
                    .collect();

                TraitItem(Box::new(Trait {
                    def_id,
                    items,
                    generics: clean_generics(generics, cx),
                    bounds: bounds.iter().filter_map(|x| clean_generic_bound(x, cx)).collect(),
                }))
            }
            ItemKind::ExternCrate(orig_name) => {
                return clean_extern_crate(item, name, orig_name, cx);
            }
            ItemKind::Use(path, kind) => {
                return clean_use_statement(item, name, path, kind, cx, &mut FxHashSet::default());
            }
            _ => unreachable!("not yet converted"),
        };

        vec![Item::from_def_id_and_parts(def_id, Some(name), kind, cx)]
    })
}

fn clean_variant<'tcx>(variant: &hir::Variant<'tcx>, cx: &mut DocContext<'tcx>) -> Item {
    let kind = VariantItem(clean_variant_data(&variant.data, &variant.disr_expr, cx));
    let what_rustc_thinks =
        Item::from_hir_id_and_parts(variant.id, Some(variant.ident.name), kind, cx);
    // don't show `pub` for variants, which are always public
    Item { visibility: Inherited, ..what_rustc_thinks }
}

fn clean_impl<'tcx>(
    impl_: &hir::Impl<'tcx>,
    hir_id: hir::HirId,
    cx: &mut DocContext<'tcx>,
) -> Vec<Item> {
    let tcx = cx.tcx;
    let mut ret = Vec::new();
    let trait_ = impl_.of_trait.as_ref().map(|t| clean_trait_ref(t, cx));
    let items = impl_
        .items
        .iter()
        .map(|ii| clean_impl_item(tcx.hir().impl_item(ii.id), cx))
        .collect::<Vec<_>>();
    let def_id = tcx.hir().local_def_id(hir_id);

    // If this impl block is an implementation of the Deref trait, then we
    // need to try inlining the target's inherent impl blocks as well.
    if trait_.as_ref().map(|t| t.def_id()) == tcx.lang_items().deref_trait() {
        build_deref_target_impls(cx, &items, &mut ret);
    }

    let for_ = clean_ty(impl_.self_ty, cx);
    let type_alias = for_.def_id(&cx.cache).and_then(|did| match tcx.def_kind(did) {
        DefKind::TyAlias => Some(clean_middle_ty(tcx.type_of(did), cx, Some(did))),
        _ => None,
    });
    let mut make_item = |trait_: Option<Path>, for_: Type, items: Vec<Item>| {
        let kind = ImplItem(Box::new(Impl {
            unsafety: impl_.unsafety,
            generics: clean_generics(impl_.generics, cx),
            trait_,
            for_,
            items,
            polarity: tcx.impl_polarity(def_id),
            kind: if utils::has_doc_flag(tcx, def_id.to_def_id(), sym::fake_variadic) {
                ImplKind::FakeVaradic
            } else {
                ImplKind::Normal
            },
        }));
        Item::from_hir_id_and_parts(hir_id, None, kind, cx)
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
    let cnum = cx.tcx.extern_mod_stmt_cnum(krate.def_id).unwrap_or(LOCAL_CRATE);
    // this is the ID of the crate itself
    let crate_def_id = cnum.as_def_id();
    let attrs = cx.tcx.hir().attrs(krate.hir_id());
    let ty_vis = cx.tcx.visibility(krate.def_id);
    let please_inline = ty_vis.is_public()
        && attrs.iter().any(|a| {
            a.has_name(sym::doc)
                && match a.meta_item_list() {
                    Some(l) => attr::list_contains_name(&l, sym::inline),
                    None => false,
                }
        });

    if please_inline {
        let mut visited = FxHashSet::default();

        let res = Res::Def(DefKind::Mod, crate_def_id);

        if let Some(items) = inline::try_inline(
            cx,
            cx.tcx.parent_module(krate.hir_id()).to_def_id(),
            Some(krate.def_id.to_def_id()),
            res,
            name,
            Some(attrs),
            &mut visited,
        ) {
            return items;
        }
    }

    // FIXME: using `from_def_id_and_kind` breaks `rustdoc/masked` for some reason
    vec![Item {
        name: Some(name),
        attrs: Box::new(Attributes::from_ast(attrs)),
        item_id: crate_def_id.into(),
        visibility: clean_visibility(ty_vis),
        kind: Box::new(ExternCrateItem { src: orig_name }),
        cfg: attrs.cfg(cx.tcx, &cx.cache.hidden_cfg),
    }]
}

fn clean_use_statement<'tcx>(
    import: &hir::Item<'tcx>,
    name: Symbol,
    path: &hir::Path<'tcx>,
    kind: hir::UseKind,
    cx: &mut DocContext<'tcx>,
    inlined_names: &mut FxHashSet<(ItemType, Symbol)>,
) -> Vec<Item> {
    // We need this comparison because some imports (for std types for example)
    // are "inserted" as well but directly by the compiler and they should not be
    // taken into account.
    if import.span.ctxt().outer_expn_data().kind == ExpnKind::AstPass(AstPass::StdImports) {
        return Vec::new();
    }

    let visibility = cx.tcx.visibility(import.def_id);
    let attrs = cx.tcx.hir().attrs(import.hir_id());
    let inline_attr = attrs.lists(sym::doc).get_word_attr(sym::inline);
    let pub_underscore = visibility.is_public() && name == kw::Underscore;
    let current_mod = cx.tcx.parent_module_from_def_id(import.def_id);

    // The parent of the module in which this import resides. This
    // is the same as `current_mod` if that's already the top
    // level module.
    let parent_mod = cx.tcx.parent_module_from_def_id(current_mod);

    // This checks if the import can be seen from a higher level module.
    // In other words, it checks if the visibility is the equivalent of
    // `pub(super)` or higher. If the current module is the top level
    // module, there isn't really a parent module, which makes the results
    // meaningless. In this case, we make sure the answer is `false`.
    let is_visible_from_parent_mod =
        visibility.is_accessible_from(parent_mod, cx.tcx) && !current_mod.is_top_level_module();

    if pub_underscore {
        if let Some(ref inline) = inline_attr {
            rustc_errors::struct_span_err!(
                cx.tcx.sess,
                inline.span(),
                E0780,
                "anonymous imports cannot be inlined"
            )
            .span_label(import.span, "anonymous import")
            .emit();
        }
    }

    // We consider inlining the documentation of `pub use` statements, but we
    // forcefully don't inline if this is not public or if the
    // #[doc(no_inline)] attribute is present.
    // Don't inline doc(hidden) imports so they can be stripped at a later stage.
    let mut denied = cx.output_format.is_json()
        || !(visibility.is_public()
            || (cx.render_options.document_private && is_visible_from_parent_mod))
        || pub_underscore
        || attrs.iter().any(|a| {
            a.has_name(sym::doc)
                && match a.meta_item_list() {
                    Some(l) => {
                        attr::list_contains_name(&l, sym::no_inline)
                            || attr::list_contains_name(&l, sym::hidden)
                    }
                    None => false,
                }
        });

    // Also check whether imports were asked to be inlined, in case we're trying to re-export a
    // crate in Rust 2018+
    let path = clean_path(path, cx);
    let inner = if kind == hir::UseKind::Glob {
        if !denied {
            let mut visited = FxHashSet::default();
            if let Some(items) = inline::try_inline_glob(cx, path.res, &mut visited, inlined_names)
            {
                return items;
            }
        }
        Import::new_glob(resolve_use_source(cx, path), true)
    } else {
        if inline_attr.is_none() {
            if let Res::Def(DefKind::Mod, did) = path.res {
                if !did.is_local() && did.is_crate_root() {
                    // if we're `pub use`ing an extern crate root, don't inline it unless we
                    // were specifically asked for it
                    denied = true;
                }
            }
        }
        if !denied {
            let mut visited = FxHashSet::default();
            let import_def_id = import.def_id.to_def_id();

            if let Some(mut items) = inline::try_inline(
                cx,
                cx.tcx.parent_module(import.hir_id()).to_def_id(),
                Some(import_def_id),
                path.res,
                name,
                Some(attrs),
                &mut visited,
            ) {
                items.push(Item::from_def_id_and_parts(
                    import_def_id,
                    None,
                    ImportItem(Import::new_simple(name, resolve_use_source(cx, path), false)),
                    cx,
                ));
                return items;
            }
        }
        Import::new_simple(name, resolve_use_source(cx, path), true)
    };

    vec![Item::from_def_id_and_parts(import.def_id.to_def_id(), None, ImportItem(inner), cx)]
}

fn clean_maybe_renamed_foreign_item<'tcx>(
    cx: &mut DocContext<'tcx>,
    item: &hir::ForeignItem<'tcx>,
    renamed: Option<Symbol>,
) -> Item {
    let def_id = item.def_id.to_def_id();
    cx.with_param_env(def_id, |cx| {
        let kind = match item.kind {
            hir::ForeignItemKind::Fn(decl, names, generics) => {
                let (generics, decl) = enter_impl_trait(cx, |cx| {
                    // NOTE: generics must be cleaned before args
                    let generics = clean_generics(generics, cx);
                    let args = clean_args_from_types_and_names(cx, decl.inputs, names);
                    let decl = clean_fn_decl_with_args(cx, decl, args);
                    (generics, decl)
                });
                ForeignFunctionItem(Box::new(Function { decl, generics }))
            }
            hir::ForeignItemKind::Static(ty, mutability) => {
                ForeignStaticItem(Static { type_: clean_ty(ty, cx), mutability, expr: None })
            }
            hir::ForeignItemKind::Type => ForeignTypeItem,
        };

        Item::from_hir_id_and_parts(
            item.hir_id(),
            Some(renamed.unwrap_or(item.ident.name)),
            kind,
            cx,
        )
    })
}

fn clean_type_binding<'tcx>(
    type_binding: &hir::TypeBinding<'tcx>,
    cx: &mut DocContext<'tcx>,
) -> TypeBinding {
    TypeBinding {
        assoc: PathSegment {
            name: type_binding.ident.name,
            args: clean_generic_args(type_binding.gen_args, cx),
        },
        kind: match type_binding.kind {
            hir::TypeBindingKind::Equality { ref term } => {
                TypeBindingKind::Equality { term: clean_hir_term(term, cx) }
            }
            hir::TypeBindingKind::Constraint { bounds } => TypeBindingKind::Constraint {
                bounds: bounds.iter().filter_map(|b| clean_generic_bound(b, cx)).collect(),
            },
        },
    }
}
