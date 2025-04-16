use rustc_data_structures::fx::{FxIndexMap, FxIndexSet, IndexEntry};
use rustc_hir as hir;
use rustc_infer::infer::region_constraints::{Constraint, RegionConstraintData};
use rustc_middle::bug;
use rustc_middle::ty::{self, Region, Ty, fold_regions};
use rustc_span::def_id::DefId;
use rustc_span::symbol::{Symbol, kw};
use rustc_trait_selection::traits::auto_trait::{self, RegionTarget};
use thin_vec::ThinVec;
use tracing::{debug, instrument};

use crate::clean::{
    self, Lifetime, clean_generic_param_def, clean_middle_ty, clean_predicate,
    clean_trait_ref_with_constraints, clean_ty_generics, simplify,
};
use crate::core::DocContext;

#[instrument(level = "debug", skip(cx))]
pub(crate) fn synthesize_auto_trait_impls<'tcx>(
    cx: &mut DocContext<'tcx>,
    item_def_id: DefId,
) -> Vec<clean::Item> {
    let tcx = cx.tcx;
    let typing_env = ty::TypingEnv::non_body_analysis(tcx, item_def_id);
    let ty = tcx.type_of(item_def_id).instantiate_identity();

    let finder = auto_trait::AutoTraitFinder::new(tcx);
    let mut auto_trait_impls: Vec<_> = cx
        .auto_traits
        .clone()
        .into_iter()
        .filter_map(|trait_def_id| {
            synthesize_auto_trait_impl(
                cx,
                ty,
                trait_def_id,
                typing_env,
                item_def_id,
                &finder,
                DiscardPositiveImpls::No,
            )
        })
        .collect();
    // We are only interested in case the type *doesn't* implement the `Sized` trait.
    if !ty.is_sized(tcx, typing_env)
        && let Some(sized_trait_def_id) = tcx.lang_items().sized_trait()
        && let Some(impl_item) = synthesize_auto_trait_impl(
            cx,
            ty,
            sized_trait_def_id,
            typing_env,
            item_def_id,
            &finder,
            DiscardPositiveImpls::Yes,
        )
    {
        auto_trait_impls.push(impl_item);
    }
    auto_trait_impls
}

#[instrument(level = "debug", skip(cx, finder))]
fn synthesize_auto_trait_impl<'tcx>(
    cx: &mut DocContext<'tcx>,
    ty: Ty<'tcx>,
    trait_def_id: DefId,
    typing_env: ty::TypingEnv<'tcx>,
    item_def_id: DefId,
    finder: &auto_trait::AutoTraitFinder<'tcx>,
    discard_positive_impls: DiscardPositiveImpls,
) -> Option<clean::Item> {
    let tcx = cx.tcx;
    let trait_ref = ty::Binder::dummy(ty::TraitRef::new(tcx, trait_def_id, [ty]));
    if !cx.generated_synthetics.insert((ty, trait_def_id)) {
        debug!("already generated, aborting");
        return None;
    }

    let result = finder.find_auto_trait_generics(ty, typing_env, trait_def_id, |info| {
        clean_param_env(cx, item_def_id, info.full_user_env, info.region_data, info.vid_to_region)
    });

    let (generics, polarity) = match result {
        auto_trait::AutoTraitResult::PositiveImpl(generics) => {
            if let DiscardPositiveImpls::Yes = discard_positive_impls {
                return None;
            }

            (generics, ty::ImplPolarity::Positive)
        }
        auto_trait::AutoTraitResult::NegativeImpl => {
            // For negative impls, we use the generic params, but *not* the predicates,
            // from the original type. Otherwise, the displayed impl appears to be a
            // conditional negative impl, when it's really unconditional.
            //
            // For example, consider the struct Foo<T: Copy>(*mut T). Using
            // the original predicates in our impl would cause us to generate
            // `impl !Send for Foo<T: Copy>`, which makes it appear that Foo
            // implements Send where T is not copy.
            //
            // Instead, we generate `impl !Send for Foo<T>`, which better
            // expresses the fact that `Foo<T>` never implements `Send`,
            // regardless of the choice of `T`.
            let mut generics = clean_ty_generics(
                cx,
                tcx.generics_of(item_def_id),
                ty::GenericPredicates::default(),
            );
            generics.where_predicates.clear();

            (generics, ty::ImplPolarity::Negative)
        }
        auto_trait::AutoTraitResult::ExplicitImpl => return None,
    };

    Some(clean::Item {
        inner: Box::new(clean::ItemInner {
            name: None,
            attrs: Default::default(),
            stability: None,
            kind: clean::ImplItem(Box::new(clean::Impl {
                safety: hir::Safety::Safe,
                generics,
                trait_: Some(clean_trait_ref_with_constraints(cx, trait_ref, ThinVec::new())),
                for_: clean_middle_ty(ty::Binder::dummy(ty), cx, None, None),
                items: Vec::new(),
                polarity,
                kind: clean::ImplKind::Auto,
            })),
            item_id: clean::ItemId::Auto { trait_: trait_def_id, for_: item_def_id },
            cfg: None,
            inline_stmt_id: None,
        }),
    })
}

#[derive(Debug)]
enum DiscardPositiveImpls {
    Yes,
    No,
}

#[instrument(level = "debug", skip(cx, region_data, vid_to_region))]
fn clean_param_env<'tcx>(
    cx: &mut DocContext<'tcx>,
    item_def_id: DefId,
    param_env: ty::ParamEnv<'tcx>,
    region_data: RegionConstraintData<'tcx>,
    vid_to_region: FxIndexMap<ty::RegionVid, ty::Region<'tcx>>,
) -> clean::Generics {
    let tcx = cx.tcx;
    let generics = tcx.generics_of(item_def_id);

    let params: ThinVec<_> = generics
        .own_params
        .iter()
        .inspect(|param| {
            if cfg!(debug_assertions) {
                debug_assert!(!param.is_anonymous_lifetime());
                if let ty::GenericParamDefKind::Type { synthetic, .. } = param.kind {
                    debug_assert!(!synthetic && param.name != kw::SelfUpper);
                }
            }
        })
        // We're basing the generics of the synthetic auto trait impl off of the generics of the
        // implementing type. Its generic parameters may have defaults, don't copy them over:
        // Generic parameter defaults are meaningless in impls.
        .map(|param| clean_generic_param_def(param, clean::ParamDefaults::No, cx))
        .collect();

    // FIXME(#111101): Incorporate the explicit predicates of the item here...
    let item_predicates: FxIndexSet<_> =
        tcx.param_env(item_def_id).caller_bounds().iter().collect();
    let where_predicates = param_env
        .caller_bounds()
        .iter()
        // FIXME: ...which hopefully allows us to simplify this:
        .filter(|pred| {
            !item_predicates.contains(pred)
                || pred
                    .as_trait_clause()
                    .is_some_and(|pred| tcx.lang_items().sized_trait() == Some(pred.def_id()))
        })
        .map(|pred| {
            fold_regions(tcx, pred, |r, _| match r.kind() {
                // FIXME: Don't `unwrap_or`, I think we should panic if we encounter an infer var that
                // we can't map to a concrete region. However, `AutoTraitFinder` *does* leak those kinds
                // of `ReVar`s for some reason at the time of writing. See `rustdoc-ui/` tests.
                // This is in dire need of an investigation into `AutoTraitFinder`.
                ty::ReVar(vid) => vid_to_region.get(&vid).copied().unwrap_or(r),
                ty::ReEarlyParam(_) | ty::ReStatic | ty::ReBound(..) | ty::ReError(_) => r,
                // FIXME(#120606): `AutoTraitFinder` can actually leak placeholder regions which feels
                // incorrect. Needs investigation.
                ty::ReLateParam(_) | ty::RePlaceholder(_) | ty::ReErased => {
                    bug!("unexpected region kind: {r:?}")
                }
            })
        })
        .flat_map(|pred| clean_predicate(pred, cx))
        .chain(clean_region_outlives_constraints(&region_data, generics))
        .collect();

    let mut generics = clean::Generics { params, where_predicates };
    simplify::sized_bounds(cx, &mut generics);
    generics.where_predicates = simplify::where_clauses(cx, generics.where_predicates);
    generics
}

/// Clean region outlives constraints to where-predicates.
///
/// This is essentially a simplified version of `lexical_region_resolve`.
///
/// However, here we determine what *needs to be* true in order for an impl to hold.
/// `lexical_region_resolve`, along with much of the rest of the compiler, is concerned
/// with determining if a given set up constraints / predicates *are* met, given some
/// starting conditions like user-provided code.
///
/// For this reason, it's easier to perform the calculations we need on our own,
/// rather than trying to make existing inference/solver code do what we want.
fn clean_region_outlives_constraints<'tcx>(
    regions: &RegionConstraintData<'tcx>,
    generics: &'tcx ty::Generics,
) -> ThinVec<clean::WherePredicate> {
    // Our goal is to "flatten" the list of constraints by eliminating all intermediate
    // `RegionVids` (region inference variables). At the end, all constraints should be
    // between `Region`s. This gives us the information we need to create the where-predicates.
    // This flattening is done in two parts.

    let mut outlives_predicates = FxIndexMap::<_, Vec<_>>::default();
    let mut map = FxIndexMap::<RegionTarget<'_>, auto_trait::RegionDeps<'_>>::default();

    // (1)  We insert all of the constraints into a map.
    // Each `RegionTarget` (a `RegionVid` or a `Region`) maps to its smaller and larger regions.
    // Note that "larger" regions correspond to sub regions in the surface language.
    // E.g., in `'a: 'b`, `'a` is the larger region.
    for (constraint, _) in &regions.constraints {
        match *constraint {
            Constraint::VarSubVar(vid1, vid2) => {
                let deps1 = map.entry(RegionTarget::RegionVid(vid1)).or_default();
                deps1.larger.insert(RegionTarget::RegionVid(vid2));

                let deps2 = map.entry(RegionTarget::RegionVid(vid2)).or_default();
                deps2.smaller.insert(RegionTarget::RegionVid(vid1));
            }
            Constraint::RegSubVar(region, vid) => {
                let deps = map.entry(RegionTarget::RegionVid(vid)).or_default();
                deps.smaller.insert(RegionTarget::Region(region));
            }
            Constraint::VarSubReg(vid, region) => {
                let deps = map.entry(RegionTarget::RegionVid(vid)).or_default();
                deps.larger.insert(RegionTarget::Region(region));
            }
            Constraint::RegSubReg(r1, r2) => {
                // The constraint is already in the form that we want, so we're done with it
                // The desired order is [larger, smaller], so flip them.
                if early_bound_region_name(r1) != early_bound_region_name(r2) {
                    outlives_predicates
                        .entry(early_bound_region_name(r2).expect("no region_name found"))
                        .or_default()
                        .push(r1);
                }
            }
        }
    }

    // (2)  Here, we "flatten" the map one element at a time. All of the elements' sub and super
    // regions are connected to each other. For example, if we have a graph that looks like this:
    //
    //     (A, B) - C - (D, E)
    //
    // where (A, B) are sub regions, and (D,E) are super regions.
    // Then, after deleting 'C', the graph will look like this:
    //
    //             ... - A - (D, E, ...)
    //             ... - B - (D, E, ...)
    //     (A, B, ...) - D - ...
    //     (A, B, ...) - E - ...
    //
    // where '...' signifies the existing sub and super regions of an entry. When two adjacent
    // `Region`s are encountered, we've computed a final constraint, and add it to our list.
    // Since we make sure to never re-add deleted items, this process will always finish.
    while !map.is_empty() {
        let target = *map.keys().next().unwrap();
        let deps = map.swap_remove(&target).unwrap();

        for smaller in &deps.smaller {
            for larger in &deps.larger {
                match (smaller, larger) {
                    (&RegionTarget::Region(smaller), &RegionTarget::Region(larger)) => {
                        if early_bound_region_name(smaller) != early_bound_region_name(larger) {
                            outlives_predicates
                                .entry(
                                    early_bound_region_name(larger).expect("no region name found"),
                                )
                                .or_default()
                                .push(smaller)
                        }
                    }
                    (&RegionTarget::RegionVid(_), &RegionTarget::Region(_)) => {
                        if let IndexEntry::Occupied(v) = map.entry(*smaller) {
                            let smaller_deps = v.into_mut();
                            smaller_deps.larger.insert(*larger);
                            smaller_deps.larger.swap_remove(&target);
                        }
                    }
                    (&RegionTarget::Region(_), &RegionTarget::RegionVid(_)) => {
                        if let IndexEntry::Occupied(v) = map.entry(*larger) {
                            let deps = v.into_mut();
                            deps.smaller.insert(*smaller);
                            deps.smaller.swap_remove(&target);
                        }
                    }
                    (&RegionTarget::RegionVid(_), &RegionTarget::RegionVid(_)) => {
                        if let IndexEntry::Occupied(v) = map.entry(*smaller) {
                            let smaller_deps = v.into_mut();
                            smaller_deps.larger.insert(*larger);
                            smaller_deps.larger.swap_remove(&target);
                        }
                        if let IndexEntry::Occupied(v) = map.entry(*larger) {
                            let larger_deps = v.into_mut();
                            larger_deps.smaller.insert(*smaller);
                            larger_deps.smaller.swap_remove(&target);
                        }
                    }
                }
            }
        }
    }

    let region_params: FxIndexSet<_> = generics
        .own_params
        .iter()
        .filter_map(|param| match param.kind {
            ty::GenericParamDefKind::Lifetime => Some(param.name),
            _ => None,
        })
        .collect();

    region_params
        .iter()
        .filter_map(|&name| {
            let bounds: FxIndexSet<_> = outlives_predicates
                .get(&name)?
                .iter()
                .map(|&region| {
                    let lifetime = early_bound_region_name(region)
                        .inspect(|name| assert!(region_params.contains(name)))
                        .map(Lifetime)
                        .unwrap_or(Lifetime::statik());
                    clean::GenericBound::Outlives(lifetime)
                })
                .collect();
            if bounds.is_empty() {
                return None;
            }
            Some(clean::WherePredicate::RegionPredicate {
                lifetime: Lifetime(name),
                bounds: bounds.into_iter().collect(),
            })
        })
        .collect()
}

fn early_bound_region_name(region: Region<'_>) -> Option<Symbol> {
    match region.kind() {
        ty::ReEarlyParam(r) => Some(r.name),
        _ => None,
    }
}
