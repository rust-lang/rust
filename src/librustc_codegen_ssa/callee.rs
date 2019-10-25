use crate::traits::*;
use rustc::ty;
use rustc::ty::subst::SubstsRef;
use rustc::hir::def_id::DefId;

pub fn resolve_and_get_fn<'tcx, Cx: CodegenMethods<'tcx>>(
    cx: &Cx,
    def_id: DefId,
    substs: SubstsRef<'tcx>,
) -> Cx::Value {
    cx.get_fn(
        ty::Instance::resolve(
            cx.tcx(),
            ty::ParamEnv::reveal_all(),
            def_id,
            substs
        ).unwrap()
    )
}

pub fn resolve_and_get_fn_for_vtable<'tcx,
    Cx: Backend<'tcx> + MiscMethods<'tcx> + TypeMethods<'tcx>
>(
    cx: &Cx,
    def_id: DefId,
    substs: SubstsRef<'tcx>,
) -> Cx::Value {
    cx.get_fn(
        ty::Instance::resolve_for_vtable(
            cx.tcx(),
            ty::ParamEnv::reveal_all(),
            def_id,
            substs
        ).unwrap()
    )
}
