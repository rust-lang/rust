use rustc_data_structures::fx::FxIndexMap;
use rustc_hir::def_id::LocalDefId;
use rustc_middle::ty::{OpaqueHiddenType, Ty, TyCtxt};

/// The shared context used by both the root as well as all its nested
/// items.
pub(super) struct BorrowCheckRootCtxt<'tcx> {
    root_def_id: LocalDefId,
    concrete_opaque_types: FxIndexMap<LocalDefId, OpaqueHiddenType<'tcx>>,
    nested_bodies: FxHashMap<LocalDefId, BorrowCheckResult<'tcx>>,
    tainted_by_errors: Option<ErrorGuaranteed>,
}

impl<'tcx> BorrowCheckRootCtxt<'tcx> {
    pub(super) fn new(root_def_id: LocalDefId) -> BorrowCheckRootCtxt<'tcx> {
        BorrowCheckRootCtxt {
            root_def_id,
            concrete_opaque_types: Default::default(),
            nested_bodies: Default::default(),
            tainted_by_errors: None,
        }
    }

    pub(super) fn add_concrete_opaque_type(
        &mut self,
        tcx: TyCtxt<'tcx>,
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
                        prev.build_mismatch_error(&hidden_ty, tcx).map(|d| d.emit());
                    e
                });
                prev.ty = Ty::new_error(tcx, guar);
            }
            // Pick a better span if there is one.
            // FIXME(oli-obk): collect multiple spans for better diagnostics down the road.
            prev.span = prev.span.substitute_dummy(hidden_ty.span);
        } else {
            self.concrete_opaque_types.0.insert(def_id, hidden_ty);
        }
    }

    pub(super) fn set_tainted_by_errors(&mut self, guar: ErrorGuaranteed) {
        self.tainted_by_errors = Some(guar);
    }

    pub(super) fn get_or_insert_nested(
        &mut self,
        tcx: TyCtxt<'tcx>,
        def_id: LocalDefId,
    ) -> &BorrowCheckResult<'tcx> {
        debug_assert_eq!(tcx.typeck_root_def_id(def_id.to_def_id()), self.root_def_id.to_def_id());
        if !self.nested_bodies.contains_key(&def_id) {
            let result = self.do_mir_borrowck(tcx, def_id, None).0;
            if let Some(prev) = self.nested_bodies.insert(def_id, result) {
                bug!("unexpected previous nested body: {prev:?}");
            }
        }

        self.nested_bodies.get(&def_id).unwrap()
    }

    pub(super) fn closure_requirements(
        &mut self,
        tcx: TyCtxt<'tcx>,
        nested_body_def_id: LocalDefId,
    ) -> &Option<ClosureRegionRequirements<'tcx>> {
        &self.get_or_insert_nested(tcx, nested_body_def_id).closure_requirements
    }

    pub(super) fn used_mut_upvars(
        &mut self,
        tcx: TyCtxt<'tcx>,
        nested_body_def_id: LocalDefId,
    ) -> SmallVec<[FieldIdx; 8]> {
        self.get_or_insert_nested(tcx, nested_body_def_id).used_mut_upvars.clone()
    }
}
