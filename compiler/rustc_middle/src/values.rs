use rustc_middle::ty::{self, AdtSizedConstraint, Ty, TyCtxt};
use rustc_query_system::Value;

impl<'tcx> Value<TyCtxt<'tcx>> for Ty<'_> {
    fn from_cycle_error(tcx: TyCtxt<'tcx>) -> Self {
        // SAFETY: This is never called when `Self` is not `Ty<'tcx>`.
        // FIXME: Represent the above fact in the trait system somehow.
        unsafe { std::mem::transmute::<Ty<'tcx>, Ty<'_>>(tcx.ty_error()) }
    }
}

impl<'tcx> Value<TyCtxt<'tcx>> for ty::SymbolName<'_> {
    fn from_cycle_error(tcx: TyCtxt<'tcx>) -> Self {
        // SAFETY: This is never called when `Self` is not `SymbolName<'tcx>`.
        // FIXME: Represent the above fact in the trait system somehow.
        unsafe {
            std::mem::transmute::<ty::SymbolName<'tcx>, ty::SymbolName<'_>>(ty::SymbolName::new(
                tcx, "<error>",
            ))
        }
    }
}

impl<'tcx> Value<TyCtxt<'tcx>> for AdtSizedConstraint<'_> {
    fn from_cycle_error(tcx: TyCtxt<'tcx>) -> Self {
        // SAFETY: This is never called when `Self` is not `AdtSizedConstraint<'tcx>`.
        // FIXME: Represent the above fact in the trait system somehow.
        unsafe {
            std::mem::transmute::<AdtSizedConstraint<'tcx>, AdtSizedConstraint<'_>>(
                AdtSizedConstraint(tcx.intern_type_list(&[tcx.ty_error()])),
            )
        }
    }
}

impl<'tcx> Value<TyCtxt<'tcx>> for ty::Binder<'_, ty::FnSig<'_>> {
    fn from_cycle_error(tcx: TyCtxt<'tcx>) -> Self {
        let err = tcx.ty_error();
        // FIXME(compiler-errors): It would be nice if we could get the
        // query key, so we could at least generate a fn signature that
        // has the right arity.
        let fn_sig = ty::Binder::dummy(tcx.mk_fn_sig(
            [].into_iter(),
            err,
            false,
            rustc_hir::Unsafety::Normal,
            rustc_target::spec::abi::Abi::Rust,
        ));

        // SAFETY: This is never called when `Self` is not `ty::Binder<'tcx, ty::FnSig<'tcx>>`.
        // FIXME: Represent the above fact in the trait system somehow.
        unsafe { std::mem::transmute::<ty::PolyFnSig<'tcx>, ty::Binder<'_, ty::FnSig<'_>>>(fn_sig) }
    }
}
