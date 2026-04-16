use rustc_middle::ty::TyCtxt;

pub(crate) fn hir_crate_items(tcx: TyCtxt<'_>) {
    tcx.force_delayed_owners_lowering();
}
