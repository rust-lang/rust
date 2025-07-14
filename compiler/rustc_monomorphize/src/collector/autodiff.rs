use rustc_middle::bug;
use rustc_middle::ty::{self, IntrinsicDef, TyCtxt};
use tracing::debug;

use crate::collector::{MonoItems, create_fn_mono_item};

pub(crate) fn collect_enzyme_autodiff_source_fn<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: ty::Instance<'tcx>,
    intrinsic: IntrinsicDef,
    output: &mut MonoItems<'tcx>,
) {
    if intrinsic.name != rustc_span::sym::enzyme_autodiff {
        return;
    };

    debug!("enzyme_autodiff found");
    let (primal, span) = match instance.args[0].kind() {
        rustc_middle::infer::canonical::ir::GenericArgKind::Type(ty) => match ty.kind() {
            ty::FnDef(def_id, substs) => {
                let span = tcx.def_span(def_id);
                let instance = ty::Instance::expect_resolve(
                    tcx,
                    ty::TypingEnv::non_body_analysis(tcx, def_id),
                    *def_id,
                    substs,
                    span,
                );

                (instance, span)
            }
            _ => bug!("expected function"),
        },
        _ => bug!("expected type"),
    };

    output.push(create_fn_mono_item(tcx, primal, span));
}
