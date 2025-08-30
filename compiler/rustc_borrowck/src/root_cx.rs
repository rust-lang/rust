use std::mem;
use std::rc::Rc;

use rustc_abi::FieldIdx;
use rustc_data_structures::fx::{FxHashMap, FxIndexMap};
use rustc_hir::def_id::LocalDefId;
use rustc_middle::mir::ConstraintCategory;
use rustc_middle::ty::{self, TyCtxt};
use rustc_span::ErrorGuaranteed;
use smallvec::SmallVec;

use crate::consumers::BorrowckConsumer;
use crate::nll::compute_closure_requirements_modulo_opaques;
use crate::region_infer::opaque_types::{
    apply_computed_concrete_opaque_types, clone_and_resolve_opaque_types,
    compute_concrete_opaque_types, detect_opaque_types_added_while_handling_opaque_types,
};
use crate::type_check::{Locations, constraint_conversion};
use crate::{
    ClosureRegionRequirements, CollectRegionConstraintsResult, ConcreteOpaqueTypes,
    PropagatedBorrowCheckResults, borrowck_check_region_constraints,
    borrowck_collect_region_constraints,
};

/// The shared context used by both the root as well as all its nested
/// items.
pub(super) struct BorrowCheckRootCtxt<'tcx> {
    pub tcx: TyCtxt<'tcx>,
    root_def_id: LocalDefId,
    concrete_opaque_types: ConcreteOpaqueTypes<'tcx>,
    /// The region constraints computed by [borrowck_collect_region_constraints]. This uses
    /// an [FxIndexMap] sto guarantee that iterating over it visits nested bodies before
    /// their parents.
    collect_region_constraints_results:
        FxIndexMap<LocalDefId, CollectRegionConstraintsResult<'tcx>>,
    propagated_borrowck_results: FxHashMap<LocalDefId, PropagatedBorrowCheckResults<'tcx>>,
    tainted_by_errors: Option<ErrorGuaranteed>,
    /// This should be `None` during normal compilation. See [`crate::consumers`] for more
    /// information on how this is used.
    pub consumer: Option<BorrowckConsumer<'tcx>>,
}

impl<'tcx> BorrowCheckRootCtxt<'tcx> {
    pub(super) fn new(
        tcx: TyCtxt<'tcx>,
        root_def_id: LocalDefId,
        consumer: Option<BorrowckConsumer<'tcx>>,
    ) -> BorrowCheckRootCtxt<'tcx> {
        BorrowCheckRootCtxt {
            tcx,
            root_def_id,
            concrete_opaque_types: Default::default(),
            collect_region_constraints_results: Default::default(),
            propagated_borrowck_results: Default::default(),
            tainted_by_errors: None,
            consumer,
        }
    }

    pub(super) fn root_def_id(&self) -> LocalDefId {
        self.root_def_id
    }

    pub(super) fn set_tainted_by_errors(&mut self, guar: ErrorGuaranteed) {
        self.tainted_by_errors = Some(guar);
    }

    pub(super) fn used_mut_upvars(
        &mut self,
        nested_body_def_id: LocalDefId,
    ) -> &SmallVec<[FieldIdx; 8]> {
        &self.propagated_borrowck_results[&nested_body_def_id].used_mut_upvars
    }

    pub(super) fn finalize(self) -> Result<&'tcx ConcreteOpaqueTypes<'tcx>, ErrorGuaranteed> {
        if let Some(guar) = self.tainted_by_errors {
            Err(guar)
        } else {
            Ok(self.tcx.arena.alloc(self.concrete_opaque_types))
        }
    }

    fn handle_opaque_type_uses(&mut self) {
        let mut per_body_info = Vec::new();
        for input in self.collect_region_constraints_results.values_mut() {
            let (num_entries, opaque_types) = clone_and_resolve_opaque_types(
                &input.infcx,
                &input.universal_region_relations,
                &mut input.constraints,
            );
            input.deferred_opaque_type_errors = compute_concrete_opaque_types(
                &input.infcx,
                &input.universal_region_relations,
                &input.constraints,
                Rc::clone(&input.location_map),
                &mut self.concrete_opaque_types,
                &opaque_types,
            );
            per_body_info.push((num_entries, opaque_types));
        }

        for (input, (opaque_types_storage_num_entries, opaque_types)) in
            self.collect_region_constraints_results.values_mut().zip(per_body_info)
        {
            if input.deferred_opaque_type_errors.is_empty() {
                input.deferred_opaque_type_errors = apply_computed_concrete_opaque_types(
                    &input.infcx,
                    &input.body_owned,
                    &input.universal_region_relations.universal_regions,
                    &input.region_bound_pairs,
                    &input.known_type_outlives_obligations,
                    &mut input.constraints,
                    &mut self.concrete_opaque_types,
                    &opaque_types,
                );
            }

            detect_opaque_types_added_while_handling_opaque_types(
                &input.infcx,
                opaque_types_storage_num_entries,
            )
        }
    }

    fn apply_closure_requirements_modulo_opaques(&mut self) {
        // Start by eagerly handling deferred closure requirements where possible.
        let mut closure_requirements_modulo_opaques = FxHashMap::default();
        let collect_region_constraints_results =
            mem::take(&mut self.collect_region_constraints_results);
        for (def_id, mut input) in collect_region_constraints_results {
            let mut depends_on_opaques = input.infcx.has_opaque_types_in_storage();
            for (def_id, args, locations) in mem::take(&mut input.deferred_closure_requirements) {
                // In case the nested body does not depend on opaques, we can fetch its final
                // result. This means we won't need to consider its closure requirements again
                // after we've handled opaques.
                let closure_requirements =
                    if let Some(result) = self.propagated_borrowck_results.get(&def_id) {
                        &result.closure_requirements
                    } else {
                        depends_on_opaques = true;
                        input.deferred_closure_requirements.push((def_id, args, locations));
                        &closure_requirements_modulo_opaques[&def_id]
                    };

                Self::apply_closure_requirements(
                    &mut input,
                    closure_requirements,
                    def_id,
                    args,
                    locations,
                );
            }

            // In case the current body does depend on opaques and it's a nested body,
            // it's parent may use its closure requirements to guide its opaque type
            // inference. Compute the current closure requirements without taking
            // constraints from opaque types into consideration.
            //
            // If the given body does not depend on any opaque types, simply finish
            // borrowck.
            if depends_on_opaques {
                if def_id != self.root_def_id {
                    let req = Self::compute_closure_requirements_modulo_opaques(&input);
                    closure_requirements_modulo_opaques.insert(def_id, req);
                }
                self.collect_region_constraints_results.insert(def_id, input);
            } else {
                assert!(input.deferred_closure_requirements.is_empty());
                let result = borrowck_check_region_constraints(self, input);
                self.propagated_borrowck_results.insert(def_id, result);
            }
        }
    }

    fn compute_closure_requirements_modulo_opaques(
        input: &CollectRegionConstraintsResult<'tcx>,
    ) -> Option<ClosureRegionRequirements<'tcx>> {
        compute_closure_requirements_modulo_opaques(
            &input.infcx,
            &input.body_owned,
            Rc::clone(&input.location_map),
            &input.universal_region_relations,
            &input.constraints,
        )
    }

    fn apply_closure_requirements(
        input: &mut CollectRegionConstraintsResult<'tcx>,
        closure_requirements: &Option<ClosureRegionRequirements<'tcx>>,
        closure_def_id: LocalDefId,
        args: ty::GenericArgsRef<'tcx>,
        locations: Locations,
    ) {
        if let Some(closure_requirements) = closure_requirements {
            constraint_conversion::ConstraintConversion::new(
                &input.infcx,
                &input.universal_region_relations.universal_regions,
                &input.region_bound_pairs,
                &input.known_type_outlives_obligations,
                locations,
                input.body_owned.span,      // irrelevant; will be overridden.
                ConstraintCategory::Boring, // same as above.
                &mut input.constraints,
            )
            .apply_closure_requirements(closure_requirements, closure_def_id, args);
        }
    }

    pub(super) fn do_mir_borrowck(&mut self) {
        // The list of all bodies we need to borrowck. This first looks at
        // nested bodies, and then their parents. This means accessing e.g.
        // `used_mut_upvars` for a closure can assume that we've already
        // checked that closure.
        let all_bodies = self
            .tcx
            .nested_bodies_within(self.root_def_id)
            .iter()
            .chain(std::iter::once(self.root_def_id));
        for def_id in all_bodies {
            let result = borrowck_collect_region_constraints(self, def_id);
            self.collect_region_constraints_results.insert(def_id, result);
        }

        self.apply_closure_requirements_modulo_opaques();

        self.handle_opaque_type_uses();

        for (def_id, mut input) in mem::take(&mut self.collect_region_constraints_results) {
            for (def_id, args, locations) in mem::take(&mut input.deferred_closure_requirements) {
                Self::apply_closure_requirements(
                    &mut input,
                    &self.propagated_borrowck_results[&def_id].closure_requirements,
                    def_id,
                    args,
                    locations,
                );
            }

            let result = borrowck_check_region_constraints(self, input);
            self.propagated_borrowck_results.insert(def_id, result);
        }
    }
}
