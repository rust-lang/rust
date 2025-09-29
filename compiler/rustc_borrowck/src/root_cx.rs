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
    apply_definition_site_hidden_types, clone_and_resolve_opaque_types,
    compute_definition_site_hidden_types, detect_opaque_types_added_while_handling_opaque_types,
};
use crate::type_check::{Locations, constraint_conversion};
use crate::{
    ClosureRegionRequirements, CollectRegionConstraintsResult, DefinitionSiteHiddenTypes,
    PropagatedBorrowCheckResults, borrowck_check_region_constraints,
    borrowck_collect_region_constraints,
};

/// The shared context used by both the root as well as all its nested
/// items.
pub(super) struct BorrowCheckRootCtxt<'tcx> {
    pub tcx: TyCtxt<'tcx>,
    root_def_id: LocalDefId,
    hidden_types: DefinitionSiteHiddenTypes<'tcx>,
    /// The region constraints computed by [borrowck_collect_region_constraints]. This uses
    /// an [FxIndexMap] to guarantee that iterating over it visits nested bodies before
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
            hidden_types: Default::default(),
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

    pub(super) fn finalize(self) -> Result<&'tcx DefinitionSiteHiddenTypes<'tcx>, ErrorGuaranteed> {
        if let Some(guar) = self.tainted_by_errors {
            Err(guar)
        } else {
            Ok(self.tcx.arena.alloc(self.hidden_types))
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
            input.deferred_opaque_type_errors = compute_definition_site_hidden_types(
                &input.infcx,
                &input.universal_region_relations,
                &input.constraints,
                Rc::clone(&input.location_map),
                &mut self.hidden_types,
                &opaque_types,
            );
            per_body_info.push((num_entries, opaque_types));
        }

        for (input, (opaque_types_storage_num_entries, opaque_types)) in
            self.collect_region_constraints_results.values_mut().zip(per_body_info)
        {
            if input.deferred_opaque_type_errors.is_empty() {
                input.deferred_opaque_type_errors = apply_definition_site_hidden_types(
                    &input.infcx,
                    &input.body_owned,
                    &input.universal_region_relations.universal_regions,
                    &input.region_bound_pairs,
                    &input.known_type_outlives_obligations,
                    &mut input.constraints,
                    &mut self.hidden_types,
                    &opaque_types,
                );
            }

            detect_opaque_types_added_while_handling_opaque_types(
                &input.infcx,
                opaque_types_storage_num_entries,
            )
        }
    }

    /// Computing defining uses of opaques may depend on the propagated region
    /// requirements of nested bodies, while applying defining uses may introduce
    /// additional region requirements we need to propagate.
    ///
    /// This results in cyclic dependency. To compute the defining uses in parent
    /// bodies, we need the closure requirements of its nested bodies, but to check
    /// non-defining uses in nested bodies, we may rely on the defining uses in the
    /// parent.
    ///
    /// We handle this issue by applying closure requirements twice. Once using the
    /// region constraints from before we've handled opaque types in the nested body
    /// - which is used by the parent to handle its defining uses - and once after.
    ///
    /// As a performance optimization, we also eagerly finish borrowck for bodies
    /// which don't depend on opaque types. In this case they get removed from
    /// `collect_region_constraints_results` and the final result gets put into
    /// `propagated_borrowck_results`.
    fn apply_closure_requirements_modulo_opaques(&mut self) {
        let mut closure_requirements_modulo_opaques = FxHashMap::default();
        // We need to `mem::take` both `self.collect_region_constraints_results` and
        // `input.deferred_closure_requirements` as we otherwise can't iterate over
        // them while mutably using the containing struct.
        let collect_region_constraints_results =
            mem::take(&mut self.collect_region_constraints_results);
        // We iterate over all bodies here, visiting nested bodies before their parent.
        for (def_id, mut input) in collect_region_constraints_results {
            // A body depends on opaque types if it either has any opaque type uses itself,
            // or it has a nested body which does.
            //
            // If the current body does not depend on any opaque types, we eagerly compute
            // its final result and write it into `self.propagated_borrowck_results`. This
            // avoids having to compute its closure requirements modulo regions, as they
            // are just the same as its final closure requirements.
            let mut depends_on_opaques = input.infcx.has_opaque_types_in_storage();

            // Iterate over all nested bodies of `input`. If that nested body depends on
            // opaque types, we apply its closure requirements modulo opaques. Otherwise
            // we use the closure requirements from its final borrowck result.
            //
            // In case we've only applied the closure requirements modulo opaques, we have
            // to later apply its closure requirements considering opaques, so we put that
            // nested body back into `deferred_closure_requirements`.
            for (def_id, args, locations) in mem::take(&mut input.deferred_closure_requirements) {
                let closure_requirements = match self.propagated_borrowck_results.get(&def_id) {
                    None => {
                        depends_on_opaques = true;
                        input.deferred_closure_requirements.push((def_id, args, locations));
                        &closure_requirements_modulo_opaques[&def_id]
                    }
                    Some(result) => &result.closure_requirements,
                };

                Self::apply_closure_requirements(
                    &mut input,
                    closure_requirements,
                    def_id,
                    args,
                    locations,
                );
            }

            // In case the current body does depend on opaques and is a nested body,
            // we need to compute its closure requirements modulo opaques so that
            // we're able to use it when visiting its parent later in this function.
            //
            // If the current body does not depend on opaque types, we finish borrowck
            // and write its result into `propagated_borrowck_results`.
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

        // We now apply the closure requirements of nested bodies modulo
        // regions. In case a body does not depend on opaque types, we
        // eagerly check its region constraints and use the final closure
        // requirements.
        //
        // We eagerly finish borrowck for bodies which don't depend on
        // opaques.
        self.apply_closure_requirements_modulo_opaques();

        // We handle opaque type uses for all bodies together.
        self.handle_opaque_type_uses();

        // Now walk over all bodies which depend on opaque types and finish borrowck.
        //
        // We first apply the final closure requirements from nested bodies which also
        // depend on opaque types and then finish borrow checking the parent. Bodies
        // which don't depend on opaques have already been fully borrowchecked in
        // `apply_closure_requirements_modulo_opaques` as an optimization.
        for (def_id, mut input) in mem::take(&mut self.collect_region_constraints_results) {
            for (def_id, args, locations) in mem::take(&mut input.deferred_closure_requirements) {
                // We visit nested bodies before their parent, so we're already
                // done with nested bodies at this point.
                let closure_requirements =
                    &self.propagated_borrowck_results[&def_id].closure_requirements;
                Self::apply_closure_requirements(
                    &mut input,
                    closure_requirements,
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
