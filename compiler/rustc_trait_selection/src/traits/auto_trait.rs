//! Support code for rustdoc and external tools.
//! You really don't want to be using this unless you need to.

use super::*;

use crate::errors::UnableToConstructConstantValue;
use crate::infer::region_constraints::{Constraint, RegionConstraintData};
use crate::infer::InferCtxt;
use crate::traits::project::ProjectAndUnifyResult;
use rustc_middle::mir::interpret::ErrorHandled;
use rustc_middle::ty::fold::{ir::TypeFolder, TypeSuperFoldable};
#[cfg(not(bootstrap))]
use rustc_middle::ty::visit::TypeVisitable;
use rustc_middle::ty::{ImplPolarity, Region, RegionVid};

use rustc_data_structures::fx::{FxHashMap, FxHashSet, FxIndexSet};

use std::collections::hash_map::Entry;
use std::collections::VecDeque;
use std::iter;

// FIXME(twk): this is obviously not nice to duplicate like that
#[derive(Eq, PartialEq, Hash, Copy, Clone, Debug)]
pub enum RegionTarget<'tcx> {
    Region(Region<'tcx>),
    RegionVid(RegionVid),
}

#[derive(Default, Debug, Clone)]
pub struct RegionDeps<'tcx> {
    larger: FxIndexSet<RegionTarget<'tcx>>,
    smaller: FxIndexSet<RegionTarget<'tcx>>,
}

pub enum AutoTraitResult<A> {
    ExplicitImpl,
    PositiveImpl(A),
    NegativeImpl,
}

#[allow(dead_code)]
impl<A> AutoTraitResult<A> {
    fn is_auto(&self) -> bool {
        matches!(self, AutoTraitResult::PositiveImpl(_) | AutoTraitResult::NegativeImpl)
    }
}

pub struct AutoTraitInfo<'cx> {
    pub full_user_env: ty::ParamEnv<'cx>,
    pub region_data: RegionConstraintData<'cx>,
    pub vid_to_region: FxHashMap<ty::RegionVid, ty::Region<'cx>>,
}

pub struct AutoTraitFinder<'tcx> {
    tcx: TyCtxt<'tcx>,
}

impl<'tcx> AutoTraitFinder<'tcx> {
    pub fn new(tcx: TyCtxt<'tcx>) -> Self {
        AutoTraitFinder { tcx }
    }

    /// Makes a best effort to determine whether and under which conditions an auto trait is
    /// implemented for a type. For example, if you have
    ///
    /// ```
    /// struct Foo<T> { data: Box<T> }
    /// ```
    ///
    /// then this might return that `Foo<T>: Send` if `T: Send` (encoded in the AutoTraitResult
    /// type). The analysis attempts to account for custom impls as well as other complex cases.
    /// This result is intended for use by rustdoc and other such consumers.
    ///
    /// (Note that due to the coinductive nature of Send, the full and correct result is actually
    /// quite simple to generate. That is, when a type has no custom impl, it is Send iff its field
    /// types are all Send. So, in our example, we might have that `Foo<T>: Send` if `Box<T>: Send`.
    /// But this is often not the best way to present to the user.)
    ///
    /// Warning: The API should be considered highly unstable, and it may be refactored or removed
    /// in the future.
    pub fn find_auto_trait_generics<A>(
        &self,
        ty: Ty<'tcx>,
        orig_env: ty::ParamEnv<'tcx>,
        trait_did: DefId,
        mut auto_trait_callback: impl FnMut(AutoTraitInfo<'tcx>) -> A,
    ) -> AutoTraitResult<A> {
        let tcx = self.tcx;

        let trait_ref = tcx.mk_trait_ref(trait_did, [ty]);

        let infcx = tcx.infer_ctxt().build();
        let mut selcx = SelectionContext::new(&infcx);
        for polarity in [true, false] {
            let result = selcx.select(&Obligation::new(
                tcx,
                ObligationCause::dummy(),
                orig_env,
                ty::Binder::dummy(ty::TraitPredicate {
                    trait_ref,
                    constness: ty::BoundConstness::NotConst,
                    polarity: if polarity {
                        ImplPolarity::Positive
                    } else {
                        ImplPolarity::Negative
                    },
                }),
            ));
            if let Ok(Some(ImplSource::UserDefined(_))) = result {
                debug!(
                    "find_auto_trait_generics({:?}): \
                 manual impl found, bailing out",
                    trait_ref
                );
                // If an explicit impl exists, it always takes priority over an auto impl
                return AutoTraitResult::ExplicitImpl;
            }
        }

        let infcx = tcx.infer_ctxt().build();
        let mut fresh_preds = FxHashSet::default();

        // Due to the way projections are handled by SelectionContext, we need to run
        // evaluate_predicates twice: once on the original param env, and once on the result of
        // the first evaluate_predicates call.
        //
        // The problem is this: most of rustc, including SelectionContext and traits::project,
        // are designed to work with a concrete usage of a type (e.g., Vec<u8>
        // fn<T>() { Vec<T> }. This information will generally never change - given
        // the 'T' in fn<T>() { ... }, we'll never know anything else about 'T'.
        // If we're unable to prove that 'T' implements a particular trait, we're done -
        // there's nothing left to do but error out.
        //
        // However, synthesizing an auto trait impl works differently. Here, we start out with
        // a set of initial conditions - the ParamEnv of the struct/enum/union we're dealing
        // with - and progressively discover the conditions we need to fulfill for it to
        // implement a certain auto trait. This ends up breaking two assumptions made by trait
        // selection and projection:
        //
        // * We can always cache the result of a particular trait selection for the lifetime of
        // an InfCtxt
        // * Given a projection bound such as '<T as SomeTrait>::SomeItem = K', if 'T:
        // SomeTrait' doesn't hold, then we don't need to care about the 'SomeItem = K'
        //
        // We fix the first assumption by manually clearing out all of the InferCtxt's caches
        // in between calls to SelectionContext.select. This allows us to keep all of the
        // intermediate types we create bound to the 'tcx lifetime, rather than needing to lift
        // them between calls.
        //
        // We fix the second assumption by reprocessing the result of our first call to
        // evaluate_predicates. Using the example of '<T as SomeTrait>::SomeItem = K', our first
        // pass will pick up 'T: SomeTrait', but not 'SomeItem = K'. On our second pass,
        // traits::project will see that 'T: SomeTrait' is in our ParamEnv, allowing
        // SelectionContext to return it back to us.

        let Some((new_env, user_env)) = self.evaluate_predicates(
            &infcx,
            trait_did,
            ty,
            orig_env,
            orig_env,
            &mut fresh_preds,
        ) else {
            return AutoTraitResult::NegativeImpl;
        };

        let (full_env, full_user_env) = self
            .evaluate_predicates(&infcx, trait_did, ty, new_env, user_env, &mut fresh_preds)
            .unwrap_or_else(|| {
                panic!("Failed to fully process: {:?} {:?} {:?}", ty, trait_did, orig_env)
            });

        debug!(
            "find_auto_trait_generics({:?}): fulfilling \
             with {:?}",
            trait_ref, full_env
        );
        infcx.clear_caches();

        // At this point, we already have all of the bounds we need. FulfillmentContext is used
        // to store all of the necessary region/lifetime bounds in the InferContext, as well as
        // an additional sanity check.
        let errors =
            super::fully_solve_bound(&infcx, ObligationCause::dummy(), full_env, ty, trait_did);
        if !errors.is_empty() {
            panic!("Unable to fulfill trait {:?} for '{:?}': {:?}", trait_did, ty, errors);
        }

        infcx.process_registered_region_obligations(&Default::default(), full_env);

        let region_data =
            infcx.inner.borrow_mut().unwrap_region_constraints().region_constraint_data().clone();

        let vid_to_region = self.map_vid_to_region(&region_data);

        let info = AutoTraitInfo { full_user_env, region_data, vid_to_region };

        AutoTraitResult::PositiveImpl(auto_trait_callback(info))
    }
}

impl<'tcx> AutoTraitFinder<'tcx> {
    /// The core logic responsible for computing the bounds for our synthesized impl.
    ///
    /// To calculate the bounds, we call `SelectionContext.select` in a loop. Like
    /// `FulfillmentContext`, we recursively select the nested obligations of predicates we
    /// encounter. However, whenever we encounter an `UnimplementedError` involving a type
    /// parameter, we add it to our `ParamEnv`. Since our goal is to determine when a particular
    /// type implements an auto trait, Unimplemented errors tell us what conditions need to be met.
    ///
    /// This method ends up working somewhat similarly to `FulfillmentContext`, but with a few key
    /// differences. `FulfillmentContext` works under the assumption that it's dealing with concrete
    /// user code. According, it considers all possible ways that a `Predicate` could be met, which
    /// isn't always what we want for a synthesized impl. For example, given the predicate `T:
    /// Iterator`, `FulfillmentContext` can end up reporting an Unimplemented error for `T:
    /// IntoIterator` -- since there's an implementation of `Iterator` where `T: IntoIterator`,
    /// `FulfillmentContext` will drive `SelectionContext` to consider that impl before giving up.
    /// If we were to rely on `FulfillmentContext`s decision, we might end up synthesizing an impl
    /// like this:
    /// ```ignore (illustrative)
    /// impl<T> Send for Foo<T> where T: IntoIterator
    /// ```
    /// While it might be technically true that Foo implements Send where `T: IntoIterator`,
    /// the bound is overly restrictive - it's really only necessary that `T: Iterator`.
    ///
    /// For this reason, `evaluate_predicates` handles predicates with type variables specially.
    /// When we encounter an `Unimplemented` error for a bound such as `T: Iterator`, we immediately
    /// add it to our `ParamEnv`, and add it to our stack for recursive evaluation. When we later
    /// select it, we'll pick up any nested bounds, without ever inferring that `T: IntoIterator`
    /// needs to hold.
    ///
    /// One additional consideration is supertrait bounds. Normally, a `ParamEnv` is only ever
    /// constructed once for a given type. As part of the construction process, the `ParamEnv` will
    /// have any supertrait bounds normalized -- e.g., if we have a type `struct Foo<T: Copy>`, the
    /// `ParamEnv` will contain `T: Copy` and `T: Clone`, since `Copy: Clone`. When we construct our
    /// own `ParamEnv`, we need to do this ourselves, through `traits::elaborate_predicates`, or
    /// else `SelectionContext` will choke on the missing predicates. However, this should never
    /// show up in the final synthesized generics: we don't want our generated docs page to contain
    /// something like `T: Copy + Clone`, as that's redundant. Therefore, we keep track of a
    /// separate `user_env`, which only holds the predicates that will actually be displayed to the
    /// user.
    fn evaluate_predicates(
        &self,
        infcx: &InferCtxt<'tcx>,
        trait_did: DefId,
        ty: Ty<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        user_env: ty::ParamEnv<'tcx>,
        fresh_preds: &mut FxHashSet<ty::Predicate<'tcx>>,
    ) -> Option<(ty::ParamEnv<'tcx>, ty::ParamEnv<'tcx>)> {
        let tcx = infcx.tcx;

        // Don't try to process any nested obligations involving predicates
        // that are already in the `ParamEnv` (modulo regions): we already
        // know that they must hold.
        for predicate in param_env.caller_bounds() {
            fresh_preds.insert(self.clean_pred(infcx, predicate));
        }

        let mut select = SelectionContext::new(&infcx);

        let mut already_visited = FxHashSet::default();
        let mut predicates = VecDeque::new();
        predicates.push_back(ty::Binder::dummy(ty::TraitPredicate {
            trait_ref: infcx.tcx.mk_trait_ref(trait_did, [ty]),

            constness: ty::BoundConstness::NotConst,
            // Auto traits are positive
            polarity: ty::ImplPolarity::Positive,
        }));

        let computed_preds = param_env.caller_bounds().iter();
        let mut user_computed_preds: FxIndexSet<_> = user_env.caller_bounds().iter().collect();

        let mut new_env = param_env;
        let dummy_cause = ObligationCause::dummy();

        while let Some(pred) = predicates.pop_front() {
            infcx.clear_caches();

            if !already_visited.insert(pred) {
                continue;
            }

            // Call `infcx.resolve_vars_if_possible` to see if we can
            // get rid of any inference variables.
            let obligation = infcx.resolve_vars_if_possible(Obligation::new(
                tcx,
                dummy_cause.clone(),
                new_env,
                pred,
            ));
            let result = select.select(&obligation);

            match result {
                Ok(Some(ref impl_source)) => {
                    // If we see an explicit negative impl (e.g., `impl !Send for MyStruct`),
                    // we immediately bail out, since it's impossible for us to continue.

                    if let ImplSource::UserDefined(ImplSourceUserDefinedData {
                        impl_def_id, ..
                    }) = impl_source
                    {
                        // Blame 'tidy' for the weird bracket placement.
                        if infcx.tcx.impl_polarity(*impl_def_id) == ty::ImplPolarity::Negative {
                            debug!(
                                "evaluate_nested_obligations: found explicit negative impl\
                                        {:?}, bailing out",
                                impl_def_id
                            );
                            return None;
                        }
                    }

                    let obligations = impl_source.borrow_nested_obligations().iter().cloned();

                    if !self.evaluate_nested_obligations(
                        ty,
                        obligations,
                        &mut user_computed_preds,
                        fresh_preds,
                        &mut predicates,
                        &mut select,
                    ) {
                        return None;
                    }
                }
                Ok(None) => {}
                Err(SelectionError::Unimplemented) => {
                    if self.is_param_no_infer(pred.skip_binder().trait_ref.substs) {
                        already_visited.remove(&pred);
                        self.add_user_pred(&mut user_computed_preds, pred.to_predicate(self.tcx));
                        predicates.push_back(pred);
                    } else {
                        debug!(
                            "evaluate_nested_obligations: `Unimplemented` found, bailing: \
                             {:?} {:?} {:?}",
                            ty,
                            pred,
                            pred.skip_binder().trait_ref.substs
                        );
                        return None;
                    }
                }
                _ => panic!("Unexpected error for '{:?}': {:?}", ty, result),
            };

            let normalized_preds = elaborate_predicates(
                tcx,
                computed_preds.clone().chain(user_computed_preds.iter().cloned()),
            )
            .map(|o| o.predicate);
            new_env = ty::ParamEnv::new(
                tcx.mk_predicates(normalized_preds),
                param_env.reveal(),
                param_env.constness(),
            );
        }

        let final_user_env = ty::ParamEnv::new(
            tcx.mk_predicates(user_computed_preds.into_iter()),
            user_env.reveal(),
            user_env.constness(),
        );
        debug!(
            "evaluate_nested_obligations(ty={:?}, trait_did={:?}): succeeded with '{:?}' \
             '{:?}'",
            ty, trait_did, new_env, final_user_env
        );

        Some((new_env, final_user_env))
    }

    /// This method is designed to work around the following issue:
    /// When we compute auto trait bounds, we repeatedly call `SelectionContext.select`,
    /// progressively building a `ParamEnv` based on the results we get.
    /// However, our usage of `SelectionContext` differs from its normal use within the compiler,
    /// in that we capture and re-reprocess predicates from `Unimplemented` errors.
    ///
    /// This can lead to a corner case when dealing with region parameters.
    /// During our selection loop in `evaluate_predicates`, we might end up with
    /// two trait predicates that differ only in their region parameters:
    /// one containing a HRTB lifetime parameter, and one containing a 'normal'
    /// lifetime parameter. For example:
    /// ```ignore (illustrative)
    /// T as MyTrait<'a>
    /// T as MyTrait<'static>
    /// ```
    /// If we put both of these predicates in our computed `ParamEnv`, we'll
    /// confuse `SelectionContext`, since it will (correctly) view both as being applicable.
    ///
    /// To solve this, we pick the 'more strict' lifetime bound -- i.e., the HRTB
    /// Our end goal is to generate a user-visible description of the conditions
    /// under which a type implements an auto trait. A trait predicate involving
    /// a HRTB means that the type needs to work with any choice of lifetime,
    /// not just one specific lifetime (e.g., `'static`).
    fn add_user_pred(
        &self,
        user_computed_preds: &mut FxIndexSet<ty::Predicate<'tcx>>,
        new_pred: ty::Predicate<'tcx>,
    ) {
        let mut should_add_new = true;
        user_computed_preds.retain(|&old_pred| {
            if let (
                ty::PredicateKind::Clause(ty::Clause::Trait(new_trait)),
                ty::PredicateKind::Clause(ty::Clause::Trait(old_trait)),
            ) = (new_pred.kind().skip_binder(), old_pred.kind().skip_binder())
            {
                if new_trait.def_id() == old_trait.def_id() {
                    let new_substs = new_trait.trait_ref.substs;
                    let old_substs = old_trait.trait_ref.substs;

                    if !new_substs.types().eq(old_substs.types()) {
                        // We can't compare lifetimes if the types are different,
                        // so skip checking `old_pred`.
                        return true;
                    }

                    for (new_region, old_region) in
                        iter::zip(new_substs.regions(), old_substs.regions())
                    {
                        match (*new_region, *old_region) {
                            // If both predicates have an `ReLateBound` (a HRTB) in the
                            // same spot, we do nothing.
                            (ty::ReLateBound(_, _), ty::ReLateBound(_, _)) => {}

                            (ty::ReLateBound(_, _), _) | (_, ty::ReVar(_)) => {
                                // One of these is true:
                                // The new predicate has a HRTB in a spot where the old
                                // predicate does not (if they both had a HRTB, the previous
                                // match arm would have executed). A HRBT is a 'stricter'
                                // bound than anything else, so we want to keep the newer
                                // predicate (with the HRBT) in place of the old predicate.
                                //
                                // OR
                                //
                                // The old predicate has a region variable where the new
                                // predicate has some other kind of region. An region
                                // variable isn't something we can actually display to a user,
                                // so we choose their new predicate (which doesn't have a region
                                // variable).
                                //
                                // In both cases, we want to remove the old predicate,
                                // from `user_computed_preds`, and replace it with the new
                                // one. Having both the old and the new
                                // predicate in a `ParamEnv` would confuse `SelectionContext`.
                                //
                                // We're currently in the predicate passed to 'retain',
                                // so we return `false` to remove the old predicate from
                                // `user_computed_preds`.
                                return false;
                            }
                            (_, ty::ReLateBound(_, _)) | (ty::ReVar(_), _) => {
                                // This is the opposite situation as the previous arm.
                                // One of these is true:
                                //
                                // The old predicate has a HRTB lifetime in a place where the
                                // new predicate does not.
                                //
                                // OR
                                //
                                // The new predicate has a region variable where the old
                                // predicate has some other type of region.
                                //
                                // We want to leave the old
                                // predicate in `user_computed_preds`, and skip adding
                                // new_pred to `user_computed_params`.
                                should_add_new = false
                            }
                            _ => {}
                        }
                    }
                }
            }
            true
        });

        if should_add_new {
            user_computed_preds.insert(new_pred);
        }
    }

    /// This is very similar to `handle_lifetimes`. However, instead of matching `ty::Region`s
    /// to each other, we match `ty::RegionVid`s to `ty::Region`s.
    fn map_vid_to_region<'cx>(
        &self,
        regions: &RegionConstraintData<'cx>,
    ) -> FxHashMap<ty::RegionVid, ty::Region<'cx>> {
        let mut vid_map: FxHashMap<RegionTarget<'cx>, RegionDeps<'cx>> = FxHashMap::default();
        let mut finished_map = FxHashMap::default();

        for constraint in regions.constraints.keys() {
            match constraint {
                &Constraint::VarSubVar(r1, r2) => {
                    {
                        let deps1 = vid_map.entry(RegionTarget::RegionVid(r1)).or_default();
                        deps1.larger.insert(RegionTarget::RegionVid(r2));
                    }

                    let deps2 = vid_map.entry(RegionTarget::RegionVid(r2)).or_default();
                    deps2.smaller.insert(RegionTarget::RegionVid(r1));
                }
                &Constraint::RegSubVar(region, vid) => {
                    {
                        let deps1 = vid_map.entry(RegionTarget::Region(region)).or_default();
                        deps1.larger.insert(RegionTarget::RegionVid(vid));
                    }

                    let deps2 = vid_map.entry(RegionTarget::RegionVid(vid)).or_default();
                    deps2.smaller.insert(RegionTarget::Region(region));
                }
                &Constraint::VarSubReg(vid, region) => {
                    finished_map.insert(vid, region);
                }
                &Constraint::RegSubReg(r1, r2) => {
                    {
                        let deps1 = vid_map.entry(RegionTarget::Region(r1)).or_default();
                        deps1.larger.insert(RegionTarget::Region(r2));
                    }

                    let deps2 = vid_map.entry(RegionTarget::Region(r2)).or_default();
                    deps2.smaller.insert(RegionTarget::Region(r1));
                }
            }
        }

        while !vid_map.is_empty() {
            let target = *vid_map.keys().next().expect("Keys somehow empty");
            let deps = vid_map.remove(&target).expect("Entry somehow missing");

            for smaller in deps.smaller.iter() {
                for larger in deps.larger.iter() {
                    match (smaller, larger) {
                        (&RegionTarget::Region(_), &RegionTarget::Region(_)) => {
                            if let Entry::Occupied(v) = vid_map.entry(*smaller) {
                                let smaller_deps = v.into_mut();
                                smaller_deps.larger.insert(*larger);
                                smaller_deps.larger.remove(&target);
                            }

                            if let Entry::Occupied(v) = vid_map.entry(*larger) {
                                let larger_deps = v.into_mut();
                                larger_deps.smaller.insert(*smaller);
                                larger_deps.smaller.remove(&target);
                            }
                        }
                        (&RegionTarget::RegionVid(v1), &RegionTarget::Region(r1)) => {
                            finished_map.insert(v1, r1);
                        }
                        (&RegionTarget::Region(_), &RegionTarget::RegionVid(_)) => {
                            // Do nothing; we don't care about regions that are smaller than vids.
                        }
                        (&RegionTarget::RegionVid(_), &RegionTarget::RegionVid(_)) => {
                            if let Entry::Occupied(v) = vid_map.entry(*smaller) {
                                let smaller_deps = v.into_mut();
                                smaller_deps.larger.insert(*larger);
                                smaller_deps.larger.remove(&target);
                            }

                            if let Entry::Occupied(v) = vid_map.entry(*larger) {
                                let larger_deps = v.into_mut();
                                larger_deps.smaller.insert(*smaller);
                                larger_deps.smaller.remove(&target);
                            }
                        }
                    }
                }
            }
        }
        finished_map
    }

    fn is_param_no_infer(&self, substs: SubstsRef<'_>) -> bool {
        self.is_of_param(substs.type_at(0)) && !substs.types().any(|t| t.has_infer_types())
    }

    pub fn is_of_param(&self, ty: Ty<'_>) -> bool {
        match ty.kind() {
            ty::Param(_) => true,
            ty::Alias(ty::Projection, p) => self.is_of_param(p.self_ty()),
            _ => false,
        }
    }

    fn is_self_referential_projection(&self, p: ty::PolyProjectionPredicate<'_>) -> bool {
        if let Some(ty) = p.term().skip_binder().ty() {
            matches!(ty.kind(), ty::Alias(ty::Projection, proj) if proj == &p.skip_binder().projection_ty)
        } else {
            false
        }
    }

    fn evaluate_nested_obligations(
        &self,
        ty: Ty<'_>,
        nested: impl Iterator<Item = Obligation<'tcx, ty::Predicate<'tcx>>>,
        computed_preds: &mut FxIndexSet<ty::Predicate<'tcx>>,
        fresh_preds: &mut FxHashSet<ty::Predicate<'tcx>>,
        predicates: &mut VecDeque<ty::PolyTraitPredicate<'tcx>>,
        selcx: &mut SelectionContext<'_, 'tcx>,
    ) -> bool {
        let dummy_cause = ObligationCause::dummy();

        for obligation in nested {
            let is_new_pred =
                fresh_preds.insert(self.clean_pred(selcx.infcx, obligation.predicate));

            // Resolve any inference variables that we can, to help selection succeed
            let predicate = selcx.infcx.resolve_vars_if_possible(obligation.predicate);

            // We only add a predicate as a user-displayable bound if
            // it involves a generic parameter, and doesn't contain
            // any inference variables.
            //
            // Displaying a bound involving a concrete type (instead of a generic
            // parameter) would be pointless, since it's always true
            // (e.g. u8: Copy)
            // Displaying an inference variable is impossible, since they're
            // an internal compiler detail without a defined visual representation
            //
            // We check this by calling is_of_param on the relevant types
            // from the various possible predicates

            let bound_predicate = predicate.kind();
            match bound_predicate.skip_binder() {
                ty::PredicateKind::Clause(ty::Clause::Trait(p)) => {
                    // Add this to `predicates` so that we end up calling `select`
                    // with it. If this predicate ends up being unimplemented,
                    // then `evaluate_predicates` will handle adding it the `ParamEnv`
                    // if possible.
                    predicates.push_back(bound_predicate.rebind(p));
                }
                ty::PredicateKind::Clause(ty::Clause::Projection(p)) => {
                    let p = bound_predicate.rebind(p);
                    debug!(
                        "evaluate_nested_obligations: examining projection predicate {:?}",
                        predicate
                    );

                    // As described above, we only want to display
                    // bounds which include a generic parameter but don't include
                    // an inference variable.
                    // Additionally, we check if we've seen this predicate before,
                    // to avoid rendering duplicate bounds to the user.
                    if self.is_param_no_infer(p.skip_binder().projection_ty.substs)
                        && !p.term().skip_binder().has_infer_types()
                        && is_new_pred
                    {
                        debug!(
                            "evaluate_nested_obligations: adding projection predicate \
                            to computed_preds: {:?}",
                            predicate
                        );

                        // Under unusual circumstances, we can end up with a self-referential
                        // projection predicate. For example:
                        // <T as MyType>::Value == <T as MyType>::Value
                        // Not only is displaying this to the user pointless,
                        // having it in the ParamEnv will cause an issue if we try to call
                        // poly_project_and_unify_type on the predicate, since this kind of
                        // predicate will normally never end up in a ParamEnv.
                        //
                        // For these reasons, we ignore these weird predicates,
                        // ensuring that we're able to properly synthesize an auto trait impl
                        if self.is_self_referential_projection(p) {
                            debug!(
                                "evaluate_nested_obligations: encountered a projection
                                 predicate equating a type with itself! Skipping"
                            );
                        } else {
                            self.add_user_pred(computed_preds, predicate);
                        }
                    }

                    // There are three possible cases when we project a predicate:
                    //
                    // 1. We encounter an error. This means that it's impossible for
                    // our current type to implement the auto trait - there's bound
                    // that we could add to our ParamEnv that would 'fix' this kind
                    // of error, as it's not caused by an unimplemented type.
                    //
                    // 2. We successfully project the predicate (Ok(Some(_))), generating
                    //  some subobligations. We then process these subobligations
                    //  like any other generated sub-obligations.
                    //
                    // 3. We receive an 'ambiguous' result (Ok(None))
                    // If we were actually trying to compile a crate,
                    // we would need to re-process this obligation later.
                    // However, all we care about is finding out what bounds
                    // are needed for our type to implement a particular auto trait.
                    // We've already added this obligation to our computed ParamEnv
                    // above (if it was necessary). Therefore, we don't need
                    // to do any further processing of the obligation.
                    //
                    // Note that we *must* try to project *all* projection predicates
                    // we encounter, even ones without inference variable.
                    // This ensures that we detect any projection errors,
                    // which indicate that our type can *never* implement the given
                    // auto trait. In that case, we will generate an explicit negative
                    // impl (e.g. 'impl !Send for MyType'). However, we don't
                    // try to process any of the generated subobligations -
                    // they contain no new information, since we already know
                    // that our type implements the projected-through trait,
                    // and can lead to weird region issues.
                    //
                    // Normally, we'll generate a negative impl as a result of encountering
                    // a type with an explicit negative impl of an auto trait
                    // (for example, raw pointers have !Send and !Sync impls)
                    // However, through some **interesting** manipulations of the type
                    // system, it's actually possible to write a type that never
                    // implements an auto trait due to a projection error, not a normal
                    // negative impl error. To properly handle this case, we need
                    // to ensure that we catch any potential projection errors,
                    // and turn them into an explicit negative impl for our type.
                    debug!("Projecting and unifying projection predicate {:?}", predicate);

                    match project::poly_project_and_unify_type(selcx, &obligation.with(self.tcx, p))
                    {
                        ProjectAndUnifyResult::MismatchedProjectionTypes(e) => {
                            debug!(
                                "evaluate_nested_obligations: Unable to unify predicate \
                                 '{:?}' '{:?}', bailing out",
                                ty, e
                            );
                            return false;
                        }
                        ProjectAndUnifyResult::Recursive => {
                            debug!("evaluate_nested_obligations: recursive projection predicate");
                            return false;
                        }
                        ProjectAndUnifyResult::Holds(v) => {
                            // We only care about sub-obligations
                            // when we started out trying to unify
                            // some inference variables. See the comment above
                            // for more information
                            if p.term().skip_binder().has_infer_types() {
                                if !self.evaluate_nested_obligations(
                                    ty,
                                    v.into_iter(),
                                    computed_preds,
                                    fresh_preds,
                                    predicates,
                                    selcx,
                                ) {
                                    return false;
                                }
                            }
                        }
                        ProjectAndUnifyResult::FailedNormalization => {
                            // It's ok not to make progress when have no inference variables -
                            // in that case, we were only performing unification to check if an
                            // error occurred (which would indicate that it's impossible for our
                            // type to implement the auto trait).
                            // However, we should always make progress (either by generating
                            // subobligations or getting an error) when we started off with
                            // inference variables
                            if p.term().skip_binder().has_infer_types() {
                                panic!("Unexpected result when selecting {:?} {:?}", ty, obligation)
                            }
                        }
                    }
                }
                ty::PredicateKind::Clause(ty::Clause::RegionOutlives(binder)) => {
                    let binder = bound_predicate.rebind(binder);
                    selcx.infcx.region_outlives_predicate(&dummy_cause, binder)
                }
                ty::PredicateKind::Clause(ty::Clause::TypeOutlives(binder)) => {
                    let binder = bound_predicate.rebind(binder);
                    match (
                        binder.no_bound_vars(),
                        binder.map_bound_ref(|pred| pred.0).no_bound_vars(),
                    ) {
                        (None, Some(t_a)) => {
                            selcx.infcx.register_region_obligation_with_cause(
                                t_a,
                                selcx.infcx.tcx.lifetimes.re_static,
                                &dummy_cause,
                            );
                        }
                        (Some(ty::OutlivesPredicate(t_a, r_b)), _) => {
                            selcx.infcx.register_region_obligation_with_cause(
                                t_a,
                                r_b,
                                &dummy_cause,
                            );
                        }
                        _ => {}
                    };
                }
                ty::PredicateKind::ConstEquate(c1, c2) => {
                    let evaluate = |c: ty::Const<'tcx>| {
                        if let ty::ConstKind::Unevaluated(unevaluated) = c.kind() {
                            match selcx.infcx.const_eval_resolve(
                                obligation.param_env,
                                unevaluated,
                                Some(obligation.cause.span),
                            ) {
                                Ok(Some(valtree)) => Ok(selcx.tcx().mk_const(valtree, c.ty())),
                                Ok(None) => {
                                    let tcx = self.tcx;
                                    let def_id = unevaluated.def.did;
                                    let reported =
                                        tcx.sess.emit_err(UnableToConstructConstantValue {
                                            span: tcx.def_span(def_id),
                                            unevaluated: unevaluated,
                                        });
                                    Err(ErrorHandled::Reported(reported))
                                }
                                Err(err) => Err(err),
                            }
                        } else {
                            Ok(c)
                        }
                    };

                    match (evaluate(c1), evaluate(c2)) {
                        (Ok(c1), Ok(c2)) => {
                            match selcx.infcx.at(&obligation.cause, obligation.param_env).eq(c1, c2)
                            {
                                Ok(_) => (),
                                Err(_) => return false,
                            }
                        }
                        _ => return false,
                    }
                }

                // There's not really much we can do with these predicates -
                // we start out with a `ParamEnv` with no inference variables,
                // and these don't correspond to adding any new bounds to
                // the `ParamEnv`.
                ty::PredicateKind::WellFormed(..)
                | ty::PredicateKind::AliasEq(..)
                | ty::PredicateKind::ObjectSafe(..)
                | ty::PredicateKind::ClosureKind(..)
                | ty::PredicateKind::Subtype(..)
                // FIXME(generic_const_exprs): you can absolutely add this as a where clauses
                | ty::PredicateKind::ConstEvaluatable(..)
                | ty::PredicateKind::Coerce(..)
                | ty::PredicateKind::TypeWellFormedFromEnv(..) => {}
                ty::PredicateKind::Ambiguous => return false,
            };
        }
        true
    }

    pub fn clean_pred(
        &self,
        infcx: &InferCtxt<'tcx>,
        p: ty::Predicate<'tcx>,
    ) -> ty::Predicate<'tcx> {
        infcx.freshen(p)
    }
}

/// Replaces all ReVars in a type with ty::Region's, using the provided map
pub struct RegionReplacer<'a, 'tcx> {
    vid_to_region: &'a FxHashMap<ty::RegionVid, ty::Region<'tcx>>,
    tcx: TyCtxt<'tcx>,
}

impl<'a, 'tcx> TypeFolder<TyCtxt<'tcx>> for RegionReplacer<'a, 'tcx> {
    fn interner(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn fold_region(&mut self, r: ty::Region<'tcx>) -> ty::Region<'tcx> {
        (match *r {
            ty::ReVar(vid) => self.vid_to_region.get(&vid).cloned(),
            _ => None,
        })
        .unwrap_or_else(|| r.super_fold_with(self))
    }
}
