use rustc_abi::FieldIdx;
use rustc_data_structures::fx::FxHashMap;
use rustc_hir::def_id::LocalDefId;
use rustc_middle::bug;
use rustc_middle::ty::{EarlyBinder, OpaqueHiddenType, Ty, TyCtxt, TypeVisitableExt};
use rustc_span::ErrorGuaranteed;
use smallvec::SmallVec;

use crate::consumers::{BodyWithBorrowckFacts, ConsumerOptions};
use crate::nll::compute_closure_requirements_modulo_opaques;
use crate::type_check::apply_closure_requirements_considering_opaques;
use crate::{
    BorrowckState, ClosureRegionRequirements, ConcreteOpaqueTypes, PropagatedBorrowCheckResults,
    resume_do_mir_borrowck, start_do_mir_borrowck,
};

/// The shared context used by both the root as well as all its nested
/// items.
pub(super) struct BorrowCheckRootCtxt<'tcx> {
    pub tcx: TyCtxt<'tcx>,
    root_def_id: LocalDefId,
    concrete_opaque_types: ConcreteOpaqueTypes<'tcx>,
    partial_results: FxHashMap<LocalDefId, Option<BorrowckState<'tcx>>>,
    closure_requirements_modulo_opaques:
        FxHashMap<LocalDefId, Option<ClosureRegionRequirements<'tcx>>>,
    final_results: FxHashMap<LocalDefId, PropagatedBorrowCheckResults<'tcx>>,
    tainted_by_errors: Option<ErrorGuaranteed>,
}

impl<'tcx> BorrowCheckRootCtxt<'tcx> {
    pub(super) fn new(tcx: TyCtxt<'tcx>, root_def_id: LocalDefId) -> BorrowCheckRootCtxt<'tcx> {
        BorrowCheckRootCtxt {
            tcx,
            root_def_id,
            concrete_opaque_types: Default::default(),
            partial_results: Default::default(),
            closure_requirements_modulo_opaques: Default::default(),
            final_results: Default::default(),
            tainted_by_errors: None,
        }
    }

    pub(super) fn root_def_id(&self) -> LocalDefId {
        self.root_def_id
    }

    /// Collect all defining uses of opaque types inside of this typeck root. This
    /// expects the hidden type to be mapped to the definition parameters of the opaque
    /// and errors if we end up with distinct hidden types.
    pub(super) fn add_concrete_opaque_type(
        &mut self,
        def_id: LocalDefId,
        hidden_ty: OpaqueHiddenType<'tcx>,
    ) {
        // Sometimes two opaque types are the same only after we remap the generic parameters
        // back to the opaque type definition. E.g. we may have `OpaqueType<X, Y>` mapped to
        // `(X, Y)` and `OpaqueType<Y, X>` mapped to `(Y, X)`, and those are the same, but we
        // only know that once we convert the generic parameters to those of the opaque type.
        if let Some(prev) = self.concrete_opaque_types.0.get_mut(&def_id) {
            if prev.ty != hidden_ty.ty {
                let guar = hidden_ty.ty.error_reported().err().unwrap_or_else(|| {
                    let (Ok(e) | Err(e)) =
                        prev.build_mismatch_error(&hidden_ty, self.tcx).map(|d| d.emit());
                    e
                });
                prev.ty = Ty::new_error(self.tcx, guar);
            }
            // Pick a better span if there is one.
            // FIXME(oli-obk): collect multiple spans for better diagnostics down the road.
            prev.span = prev.span.substitute_dummy(hidden_ty.span);
        } else {
            self.concrete_opaque_types.0.insert(def_id, hidden_ty);
        }
    }

    pub(super) fn get_concrete_opaque_type(
        &mut self,
        def_id: LocalDefId,
    ) -> Option<EarlyBinder<'tcx, OpaqueHiddenType<'tcx>>> {
        self.concrete_opaque_types.0.get(&def_id).map(|ty| EarlyBinder::bind(*ty))
    }

    pub(super) fn set_tainted_by_errors(&mut self, guar: ErrorGuaranteed) {
        self.tainted_by_errors = Some(guar);
    }

    fn compute_partial_results(&mut self, def_id: LocalDefId) {
        debug_assert_eq!(
            self.tcx.typeck_root_def_id(def_id.to_def_id()),
            self.root_def_id.to_def_id()
        );
        if !self.partial_results.contains_key(&def_id) {
            let result = start_do_mir_borrowck(self, def_id, None);
            // We only need to store the partial result if it depends on opaque types
            // or depends on a nested item which does.
            let relies_on_opaques = result.infcx.has_opaque_types_in_storage()
                || !result.deferred_closure_requirements.is_empty();

            let to_insert = if !relies_on_opaques {
                let final_result = resume_do_mir_borrowck(self, None, result).0;
                if self.final_results.insert(def_id, final_result).is_some() {
                    bug!("unexpected previous final result for {def_id:?}");
                }
                None
            } else {
                Some(result)
            };

            if self.partial_results.insert(def_id, to_insert).is_some() {
                bug!("unexpected previous partial result for: {def_id:?}")
            }
        }
    }

    fn get_or_insert_final(&mut self, def_id: LocalDefId) -> &PropagatedBorrowCheckResults<'tcx> {
        debug_assert_eq!(
            self.tcx.typeck_root_def_id(def_id.to_def_id()),
            self.root_def_id.to_def_id()
        );
        if !self.final_results.contains_key(&def_id) {
            let mut yield_do_mir_borrowck = self.partial_results.remove(&def_id).unwrap().unwrap();
            self.handle_opaque_type_uses(&mut yield_do_mir_borrowck);
            apply_closure_requirements_considering_opaques(self, &mut yield_do_mir_borrowck);
            let result = resume_do_mir_borrowck(self, None, yield_do_mir_borrowck).0;
            if let Some(prev) = self.final_results.insert(def_id, result) {
                bug!("unexpected previous nested body: {prev:?}");
            }
        }

        self.final_results.get(&def_id).unwrap()
    }

    pub(crate) fn get_closure_requirements_considering_regions(
        &mut self,
        def_id: LocalDefId,
    ) -> &Option<ClosureRegionRequirements<'tcx>> {
        &self.get_or_insert_final(def_id).closure_requirements
    }

    /// Get the closure requirements of the nested body `def_id`. In case
    /// this nested body relies on opaques, checking that the hidden type
    /// matches the final definition may introduce new region constraints
    /// we aren't considering yet.
    ///
    /// In these cases, we need to later add these requirements again, including
    /// the constraints from opaque types.
    pub(crate) fn get_closure_requirements_modulo_opaques(
        &mut self,
        def_id: LocalDefId,
    ) -> (&Option<ClosureRegionRequirements<'tcx>>, bool) {
        self.compute_partial_results(def_id);
        // In case the nested item does not use any opaque types or depend on nested items
        // which do, we eagerly compute its final result to avoid duplicate work.
        if let Some(final_result) = self.final_results.get(&def_id) {
            (&final_result.closure_requirements, false)
        } else if self.closure_requirements_modulo_opaques.contains_key(&def_id) {
            (self.closure_requirements_modulo_opaques.get(&def_id).unwrap(), true)
        } else {
            let partial_result = self.partial_results.get(&def_id).unwrap().as_ref().unwrap();
            let modulo_opaques = compute_closure_requirements_modulo_opaques(partial_result);
            self.closure_requirements_modulo_opaques.insert(def_id, modulo_opaques);
            (self.closure_requirements_modulo_opaques.get(&def_id).unwrap(), true)
        }
    }

    pub(super) fn used_mut_upvars(
        &mut self,
        nested_body_def_id: LocalDefId,
    ) -> &SmallVec<[FieldIdx; 8]> {
        &self.get_or_insert_final(nested_body_def_id).used_mut_upvars
    }

    /// The actual borrowck routine. This should only be called for the typeck root,
    /// not for any nested bodies.
    pub(super) fn borrowck_root(
        mut self,
        consumer_options: Option<ConsumerOptions>,
    ) -> (
        Result<&'tcx ConcreteOpaqueTypes<'tcx>, ErrorGuaranteed>,
        Option<Box<BodyWithBorrowckFacts<'tcx>>>,
    ) {
        let root_def_id = self.root_def_id;
        let mut yield_do_mir_borrowck =
            start_do_mir_borrowck(&mut self, root_def_id, consumer_options);

        self.handle_opaque_type_uses(&mut yield_do_mir_borrowck);
        apply_closure_requirements_considering_opaques(&mut self, &mut yield_do_mir_borrowck);
        let (PropagatedBorrowCheckResults { closure_requirements, used_mut_upvars }, consumer_data) =
            resume_do_mir_borrowck(&mut self, consumer_options, yield_do_mir_borrowck);

        // We need to manually borrowck all nested bodies from the HIR as
        // we do not generate MIR for dead code. Not doing so causes us to
        // never check closures in dead code.
        let nested_bodies = self.tcx.nested_bodies_within(root_def_id);
        for def_id in nested_bodies {
            if !self.final_results.contains_key(&def_id) {
                self.compute_partial_results(def_id);
                let _ = self.get_or_insert_final(def_id);
            }
        }

        #[allow(rustc::potential_query_instability)]
        if cfg!(debug_assertions) {
            assert!(closure_requirements.is_none());
            assert!(used_mut_upvars.is_empty());
            assert!(self.partial_results.values().all(|entry| entry.is_none()));
        }

        let result = if let Some(guar) = self.tainted_by_errors {
            Err(guar)
        } else {
            Ok(&*self.tcx.arena.alloc(self.concrete_opaque_types))
        };
        (result, consumer_data)
    }
}
