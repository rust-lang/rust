// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Support code for rustdoc and external tools . You really don't
//! want to be using this unless you need to.

use super::*;

use std::collections::hash_map::Entry;
use std::collections::VecDeque;

use rustc_data_structures::fx::{FxHashMap, FxHashSet};

use infer::region_constraints::{Constraint, RegionConstraintData};
use infer::{InferCtxt, RegionObligation};

use ty::fold::TypeFolder;
use ty::{Region, RegionVid};

// FIXME(twk): this is obviously not nice to duplicate like that
#[derive(Eq, PartialEq, Hash, Copy, Clone, Debug)]
pub enum RegionTarget<'tcx> {
    Region(Region<'tcx>),
    RegionVid(RegionVid),
}

#[derive(Default, Debug, Clone)]
pub struct RegionDeps<'tcx> {
    larger: FxHashSet<RegionTarget<'tcx>>,
    smaller: FxHashSet<RegionTarget<'tcx>>,
}

pub enum AutoTraitResult<A> {
    ExplicitImpl,
    PositiveImpl(A),
    NegativeImpl,
}

impl<A> AutoTraitResult<A> {
    fn is_auto(&self) -> bool {
        match *self {
            AutoTraitResult::PositiveImpl(_) | AutoTraitResult::NegativeImpl => true,
            _ => false,
        }
    }
}

pub struct AutoTraitInfo<'cx> {
    pub full_user_env: ty::ParamEnv<'cx>,
    pub region_data: RegionConstraintData<'cx>,
    pub names_map: FxHashSet<String>,
    pub vid_to_region: FxHashMap<ty::RegionVid, ty::Region<'cx>>,
}

pub struct AutoTraitFinder<'a, 'tcx: 'a> {
    tcx: &'a TyCtxt<'a, 'tcx, 'tcx>,
}

impl<'a, 'tcx> AutoTraitFinder<'a, 'tcx> {
    pub fn new(tcx: &'a TyCtxt<'a, 'tcx, 'tcx>) -> Self {
        AutoTraitFinder { tcx }
    }

    /// Make a best effort to determine whether and under which conditions an auto trait is
    /// implemented for a type. For example, if you have
    ///
    /// ```
    /// struct Foo<T> { data: Box<T> }
    /// ```

    /// then this might return that Foo<T>: Send if T: Send (encoded in the AutoTraitResult type).
    /// The analysis attempts to account for custom impls as well as other complex cases. This
    /// result is intended for use by rustdoc and other such consumers.

    /// (Note that due to the coinductive nature of Send, the full and correct result is actually
    /// quite simple to generate. That is, when a type has no custom impl, it is Send iff its field
    /// types are all Send. So, in our example, we might have that Foo<T>: Send if Box<T>: Send.
    /// But this is often not the best way to present to the user.)

    /// Warning: The API should be considered highly unstable, and it may be refactored or removed
    /// in the future.
    pub fn find_auto_trait_generics<A>(
        &self,
        did: DefId,
        trait_did: DefId,
        generics: &ty::Generics,
        auto_trait_callback: impl for<'i> Fn(&InferCtxt<'_, 'tcx, 'i>, AutoTraitInfo<'i>) -> A,
    ) -> AutoTraitResult<A> {
        let tcx = self.tcx;
        let ty = self.tcx.type_of(did);

        let orig_params = tcx.param_env(did);

        let trait_ref = ty::TraitRef {
            def_id: trait_did,
            substs: tcx.mk_substs_trait(ty, &[]),
        };

        let trait_pred = ty::Binder::bind(trait_ref);

        let bail_out = tcx.infer_ctxt().enter(|infcx| {
            let mut selcx = SelectionContext::with_negative(&infcx, true);
            let result = selcx.select(&Obligation::new(
                ObligationCause::dummy(),
                orig_params,
                trait_pred.to_poly_trait_predicate(),
            ));
            match result {
                Ok(Some(Vtable::VtableImpl(_))) => {
                    debug!(
                        "find_auto_trait_generics(did={:?}, trait_did={:?}, generics={:?}): \
                         manual impl found, bailing out",
                        did, trait_did, generics
                    );
                    return true;
                }
                _ => return false,
            };
        });

        // If an explicit impl exists, it always takes priority over an auto impl
        if bail_out {
            return AutoTraitResult::ExplicitImpl;
        }

        return tcx.infer_ctxt().enter(|mut infcx| {
            let mut fresh_preds = FxHashSet();

            // Due to the way projections are handled by SelectionContext, we need to run
            // evaluate_predicates twice: once on the original param env, and once on the result of
            // the first evaluate_predicates call.
            //
            // The problem is this: most of rustc, including SelectionContext and traits::project,
            // are designed to work with a concrete usage of a type (e.g. Vec<u8>
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

            let (new_env, user_env) = match self.evaluate_predicates(
                &mut infcx,
                did,
                trait_did,
                ty,
                orig_params.clone(),
                orig_params,
                &mut fresh_preds,
                false,
            ) {
                Some(e) => e,
                None => return AutoTraitResult::NegativeImpl,
            };

            let (full_env, full_user_env) = self.evaluate_predicates(
                &mut infcx,
                did,
                trait_did,
                ty,
                new_env.clone(),
                user_env,
                &mut fresh_preds,
                true,
            ).unwrap_or_else(|| {
                panic!(
                    "Failed to fully process: {:?} {:?} {:?}",
                    ty, trait_did, orig_params
                )
            });

            debug!(
                "find_auto_trait_generics(did={:?}, trait_did={:?}, generics={:?}): fulfilling \
                 with {:?}",
                did, trait_did, generics, full_env
            );
            infcx.clear_caches();

            // At this point, we already have all of the bounds we need. FulfillmentContext is used
            // to store all of the necessary region/lifetime bounds in the InferContext, as well as
            // an additional sanity check.
            let mut fulfill = FulfillmentContext::new();
            fulfill.register_bound(
                &infcx,
                full_env,
                ty,
                trait_did,
                ObligationCause::misc(DUMMY_SP, ast::DUMMY_NODE_ID),
            );
            fulfill.select_all_or_error(&infcx).unwrap_or_else(|e| {
                panic!(
                    "Unable to fulfill trait {:?} for '{:?}': {:?}",
                    trait_did, ty, e
                )
            });

            let names_map: FxHashSet<String> = generics
                .params
                .iter()
                .filter_map(|param| match param.kind {
                    ty::GenericParamDefKind::Lifetime => Some(param.name.to_string()),
                    _ => None
                })
                .collect();

            let body_ids: FxHashSet<_> = infcx
                .region_obligations
                .borrow()
                .iter()
                .map(|&(id, _)| id)
                .collect();

            for id in body_ids {
                infcx.process_registered_region_obligations(&[], None, full_env.clone(), id);
            }

            let region_data = infcx
                .borrow_region_constraints()
                .region_constraint_data()
                .clone();

            let vid_to_region = self.map_vid_to_region(&region_data);

            let info = AutoTraitInfo {
                full_user_env,
                region_data,
                names_map,
                vid_to_region,
            };

            return AutoTraitResult::PositiveImpl(auto_trait_callback(&infcx, info));
        });
    }
}

impl<'a, 'tcx> AutoTraitFinder<'a, 'tcx> {
    // The core logic responsible for computing the bounds for our synthesized impl.
    //
    // To calculate the bounds, we call SelectionContext.select in a loop. Like FulfillmentContext,
    // we recursively select the nested obligations of predicates we encounter. However, whenver we
    // encounter an UnimplementedError involving a type parameter, we add it to our ParamEnv. Since
    // our goal is to determine when a particular type implements an auto trait, Unimplemented
    // errors tell us what conditions need to be met.
    //
    // This method ends up working somewhat similary to FulfillmentContext, but with a few key
    // differences. FulfillmentContext works under the assumption that it's dealing with concrete
    // user code. According, it considers all possible ways that a Predicate could be met - which
    // isn't always what we want for a synthesized impl. For example, given the predicate 'T:
    // Iterator', FulfillmentContext can end up reporting an Unimplemented error for T:
    // IntoIterator - since there's an implementation of Iteratpr where T: IntoIterator,
    // FulfillmentContext will drive SelectionContext to consider that impl before giving up. If we
    // were to rely on FulfillmentContext's decision, we might end up synthesizing an impl like
    // this:
    // 'impl<T> Send for Foo<T> where T: IntoIterator'
    //
    // While it might be technically true that Foo implements Send where T: IntoIterator,
    // the bound is overly restrictive - it's really only necessary that T: Iterator.
    //
    // For this reason, evaluate_predicates handles predicates with type variables specially. When
    // we encounter an Unimplemented error for a bound such as 'T: Iterator', we immediately add it
    // to our ParamEnv, and add it to our stack for recursive evaluation. When we later select it,
    // we'll pick up any nested bounds, without ever inferring that 'T: IntoIterator' needs to
    // hold.
    //
    // One additonal consideration is supertrait bounds. Normally, a ParamEnv is only ever
    // consutrcted once for a given type. As part of the construction process, the ParamEnv will
    // have any supertrait bounds normalized - e.g. if we have a type 'struct Foo<T: Copy>', the
    // ParamEnv will contain 'T: Copy' and 'T: Clone', since 'Copy: Clone'. When we construct our
    // own ParamEnv, we need to do this outselves, through traits::elaborate_predicates, or else
    // SelectionContext will choke on the missing predicates. However, this should never show up in
    // the final synthesized generics: we don't want our generated docs page to contain something
    // like 'T: Copy + Clone', as that's redundant. Therefore, we keep track of a separate
    // 'user_env', which only holds the predicates that will actually be displayed to the user.
    pub fn evaluate_predicates<'b, 'gcx, 'c>(
        &self,
        infcx: &InferCtxt<'b, 'tcx, 'c>,
        ty_did: DefId,
        trait_did: DefId,
        ty: ty::Ty<'c>,
        param_env: ty::ParamEnv<'c>,
        user_env: ty::ParamEnv<'c>,
        fresh_preds: &mut FxHashSet<ty::Predicate<'c>>,
        only_projections: bool,
    ) -> Option<(ty::ParamEnv<'c>, ty::ParamEnv<'c>)> {
        let tcx = infcx.tcx;

        let mut select = SelectionContext::new(&infcx);

        let mut already_visited = FxHashSet();
        let mut predicates = VecDeque::new();
        predicates.push_back(ty::Binder::bind(ty::TraitPredicate {
            trait_ref: ty::TraitRef {
                def_id: trait_did,
                substs: infcx.tcx.mk_substs_trait(ty, &[]),
            },
        }));

        let mut computed_preds: FxHashSet<_> = param_env.caller_bounds.iter().cloned().collect();
        let mut user_computed_preds: FxHashSet<_> =
            user_env.caller_bounds.iter().cloned().collect();

        let mut new_env = param_env.clone();
        let dummy_cause = ObligationCause::misc(DUMMY_SP, ast::DUMMY_NODE_ID);

        while let Some(pred) = predicates.pop_front() {
            infcx.clear_caches();

            if !already_visited.insert(pred.clone()) {
                continue;
            }

            let result = select.select(&Obligation::new(dummy_cause.clone(), new_env, pred));

            match &result {
                &Ok(Some(ref vtable)) => {
                    let obligations = vtable.clone().nested_obligations().into_iter();

                    if !self.evaluate_nested_obligations(
                        ty,
                        obligations,
                        &mut user_computed_preds,
                        fresh_preds,
                        &mut predicates,
                        &mut select,
                        only_projections,
                    ) {
                        return None;
                    }
                }
                &Ok(None) => {}
                &Err(SelectionError::Unimplemented) => {
                    if self.is_of_param(pred.skip_binder().trait_ref.substs) {
                        already_visited.remove(&pred);
                        user_computed_preds.insert(ty::Predicate::Trait(pred.clone()));
                        predicates.push_back(pred);
                    } else {
                        debug!(
                            "evaluate_nested_obligations: Unimplemented found, bailing: \
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

            computed_preds.extend(user_computed_preds.iter().cloned());
            let normalized_preds =
                elaborate_predicates(tcx, computed_preds.clone().into_iter().collect());
            new_env = ty::ParamEnv::new(tcx.mk_predicates(normalized_preds), param_env.reveal);
        }

        let final_user_env = ty::ParamEnv::new(
            tcx.mk_predicates(user_computed_preds.into_iter()),
            user_env.reveal,
        );
        debug!(
            "evaluate_nested_obligations(ty_did={:?}, trait_did={:?}): succeeded with '{:?}' \
             '{:?}'",
            ty_did, trait_did, new_env, final_user_env
        );

        return Some((new_env, final_user_env));
    }

    pub fn region_name(&self, region: Region) -> Option<String> {
        match region {
            &ty::ReEarlyBound(r) => Some(r.name.to_string()),
            _ => None,
        }
    }

    pub fn get_lifetime(&self, region: Region, names_map: &FxHashMap<String, String>) -> String {
        self.region_name(region)
            .map(|name| {
                names_map.get(&name).unwrap_or_else(|| {
                    panic!("Missing lifetime with name {:?} for {:?}", name, region)
                })
            })
            .unwrap_or(&"'static".to_string())
            .clone()
    }

    // This is very similar to handle_lifetimes. However, instead of matching ty::Region's
    // to each other, we match ty::RegionVid's to ty::Region's
    pub fn map_vid_to_region<'cx>(
        &self,
        regions: &RegionConstraintData<'cx>,
    ) -> FxHashMap<ty::RegionVid, ty::Region<'cx>> {
        let mut vid_map: FxHashMap<RegionTarget<'cx>, RegionDeps<'cx>> = FxHashMap();
        let mut finished_map = FxHashMap();

        for constraint in regions.constraints.keys() {
            match constraint {
                &Constraint::VarSubVar(r1, r2) => {
                    {
                        let deps1 = vid_map
                            .entry(RegionTarget::RegionVid(r1))
                            .or_insert_with(|| Default::default());
                        deps1.larger.insert(RegionTarget::RegionVid(r2));
                    }

                    let deps2 = vid_map
                        .entry(RegionTarget::RegionVid(r2))
                        .or_insert_with(|| Default::default());
                    deps2.smaller.insert(RegionTarget::RegionVid(r1));
                }
                &Constraint::RegSubVar(region, vid) => {
                    {
                        let deps1 = vid_map
                            .entry(RegionTarget::Region(region))
                            .or_insert_with(|| Default::default());
                        deps1.larger.insert(RegionTarget::RegionVid(vid));
                    }

                    let deps2 = vid_map
                        .entry(RegionTarget::RegionVid(vid))
                        .or_insert_with(|| Default::default());
                    deps2.smaller.insert(RegionTarget::Region(region));
                }
                &Constraint::VarSubReg(vid, region) => {
                    finished_map.insert(vid, region);
                }
                &Constraint::RegSubReg(r1, r2) => {
                    {
                        let deps1 = vid_map
                            .entry(RegionTarget::Region(r1))
                            .or_insert_with(|| Default::default());
                        deps1.larger.insert(RegionTarget::Region(r2));
                    }

                    let deps2 = vid_map
                        .entry(RegionTarget::Region(r2))
                        .or_insert_with(|| Default::default());
                    deps2.smaller.insert(RegionTarget::Region(r1));
                }
            }
        }

        while !vid_map.is_empty() {
            let target = vid_map.keys().next().expect("Keys somehow empty").clone();
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
                            // Do nothing - we don't care about regions that are smaller than vids
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

    pub fn is_of_param(&self, substs: &Substs) -> bool {
        if substs.is_noop() {
            return false;
        }

        return match substs.type_at(0).sty {
            ty::TyParam(_) => true,
            ty::TyProjection(p) => self.is_of_param(p.substs),
            _ => false,
        };
    }

    pub fn evaluate_nested_obligations<
        'b,
        'c,
        'd,
        'cx,
        T: Iterator<Item = Obligation<'cx, ty::Predicate<'cx>>>,
    >(
        &self,
        ty: ty::Ty,
        nested: T,
        computed_preds: &'b mut FxHashSet<ty::Predicate<'cx>>,
        fresh_preds: &'b mut FxHashSet<ty::Predicate<'cx>>,
        predicates: &'b mut VecDeque<ty::PolyTraitPredicate<'cx>>,
        select: &mut SelectionContext<'c, 'd, 'cx>,
        only_projections: bool,
    ) -> bool {
        let dummy_cause = ObligationCause::misc(DUMMY_SP, ast::DUMMY_NODE_ID);

        for (obligation, predicate) in nested
            .filter(|o| o.recursion_depth == 1)
            .map(|o| (o.clone(), o.predicate.clone()))
        {
            let is_new_pred =
                fresh_preds.insert(self.clean_pred(select.infcx(), predicate.clone()));

            match &predicate {
                &ty::Predicate::Trait(ref p) => {
                    let substs = &p.skip_binder().trait_ref.substs;

                    if self.is_of_param(substs) && !only_projections && is_new_pred {
                        computed_preds.insert(predicate);
                    }
                    predicates.push_back(p.clone());
                }
                &ty::Predicate::Projection(p) => {
                    // If the projection isn't all type vars, then
                    // we don't want to add it as a bound
                    if self.is_of_param(p.skip_binder().projection_ty.substs) && is_new_pred {
                        computed_preds.insert(predicate);
                    } else {
                        match poly_project_and_unify_type(select, &obligation.with(p.clone())) {
                            Err(e) => {
                                debug!(
                                    "evaluate_nested_obligations: Unable to unify predicate \
                                     '{:?}' '{:?}', bailing out",
                                    ty, e
                                );
                                return false;
                            }
                            Ok(Some(v)) => {
                                if !self.evaluate_nested_obligations(
                                    ty,
                                    v.clone().iter().cloned(),
                                    computed_preds,
                                    fresh_preds,
                                    predicates,
                                    select,
                                    only_projections,
                                ) {
                                    return false;
                                }
                            }
                            Ok(None) => {
                                panic!("Unexpected result when selecting {:?} {:?}", ty, obligation)
                            }
                        }
                    }
                }
                &ty::Predicate::RegionOutlives(ref binder) => {
                    if let Err(_) = select
                        .infcx()
                        .region_outlives_predicate(&dummy_cause, binder)
                    {
                        return false;
                    }
                }
                &ty::Predicate::TypeOutlives(ref binder) => {
                    match (
                        binder.no_late_bound_regions(),
                        binder.map_bound_ref(|pred| pred.0).no_late_bound_regions(),
                    ) {
                        (None, Some(t_a)) => {
                            select.infcx().register_region_obligation(
                                ast::DUMMY_NODE_ID,
                                RegionObligation {
                                    sup_type: t_a,
                                    sub_region: select.infcx().tcx.types.re_static,
                                    cause: dummy_cause.clone(),
                                },
                            );
                        }
                        (Some(ty::OutlivesPredicate(t_a, r_b)), _) => {
                            select.infcx().register_region_obligation(
                                ast::DUMMY_NODE_ID,
                                RegionObligation {
                                    sup_type: t_a,
                                    sub_region: r_b,
                                    cause: dummy_cause.clone(),
                                },
                            );
                        }
                        _ => {}
                    };
                }
                _ => panic!("Unexpected predicate {:?} {:?}", ty, predicate),
            };
        }
        return true;
    }

    pub fn clean_pred<'c, 'd, 'cx>(
        &self,
        infcx: &InferCtxt<'c, 'd, 'cx>,
        p: ty::Predicate<'cx>,
    ) -> ty::Predicate<'cx> {
        infcx.freshen(p)
    }
}

// Replaces all ReVars in a type with ty::Region's, using the provided map
pub struct RegionReplacer<'a, 'gcx: 'a + 'tcx, 'tcx: 'a> {
    vid_to_region: &'a FxHashMap<ty::RegionVid, ty::Region<'tcx>>,
    tcx: TyCtxt<'a, 'gcx, 'tcx>,
}

impl<'a, 'gcx, 'tcx> TypeFolder<'gcx, 'tcx> for RegionReplacer<'a, 'gcx, 'tcx> {
    fn tcx<'b>(&'b self) -> TyCtxt<'b, 'gcx, 'tcx> {
        self.tcx
    }

    fn fold_region(&mut self, r: ty::Region<'tcx>) -> ty::Region<'tcx> {
        (match r {
            &ty::ReVar(vid) => self.vid_to_region.get(&vid).cloned(),
            _ => None,
        }).unwrap_or_else(|| r.super_fold_with(self))
    }
}
