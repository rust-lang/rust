use rustc_data_structures::fx::FxIndexMap;
use rustc_hir::def_id::LocalDefId;
use rustc_middle::ty::{OpaqueHiddenType, Ty, TyCtxt};

#[derive(Debug, Default)]
pub(super) struct ConcreteOpaqueTypes<'tcx> {
    concrete_opaque_types: FxIndexMap<LocalDefId, OpaqueHiddenType<'tcx>>,
}

impl<'tcx> ConcreteOpaqueTypes<'tcx> {
    pub(super) fn is_empty(&self) -> bool {
        self.concrete_opaque_types.is_empty()
    }

    pub(super) fn into_inner(self) -> FxIndexMap<LocalDefId, OpaqueHiddenType<'tcx>> {
        self.concrete_opaque_types
    }

    /// Insert an opaque type into the list of opaque types defined by this function
    /// after mapping the hidden type to the generic parameters of the opaque type
    /// definition.
    pub(super) fn insert(
        &mut self,
        tcx: TyCtxt<'tcx>,
        def_id: LocalDefId,
        hidden_ty: OpaqueHiddenType<'tcx>,
    ) {
        // Sometimes two opaque types are the same only after we remap the generic parameters
        // back to the opaque type definition. E.g. we may have `OpaqueType<X, Y>` mapped to
        // `(X, Y)` and `OpaqueType<Y, X>` mapped to `(Y, X)`, and those are the same, but we
        // only know that once we convert the generic parameters to those of the opaque type.
        if let Some(prev) = self.concrete_opaque_types.get_mut(&def_id) {
            if prev.ty != hidden_ty.ty {
                let (Ok(guar) | Err(guar)) =
                    prev.build_mismatch_error(&hidden_ty, tcx).map(|d| d.emit());
                prev.ty = Ty::new_error(tcx, guar);
            }
            // Pick a better span if there is one.
            // FIXME(oli-obk): collect multiple spans for better diagnostics down the road.
            prev.span = prev.span.substitute_dummy(hidden_ty.span);
        } else {
            self.concrete_opaque_types.insert(def_id, hidden_ty);
        }
    }

    pub(super) fn extend_from_nested_body(
        &mut self,
        tcx: TyCtxt<'tcx>,
        nested_body: &FxIndexMap<LocalDefId, OpaqueHiddenType<'tcx>>,
    ) {
        for (&def_id, &hidden_ty) in nested_body {
            self.insert(tcx, def_id, hidden_ty);
        }
    }
}
