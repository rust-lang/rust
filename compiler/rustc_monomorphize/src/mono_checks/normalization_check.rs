use std::ops::ControlFlow;

use rustc_middle::mir::Body;
use rustc_middle::ty::{self, Instance, Ty, TyCtxt, TypeVisitable, TypeVisitor};

use crate::errors::RecursionLimit;

struct NormalizationChecker<'tcx> {
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
}

impl<'tcx> TypeVisitor<TyCtxt<'tcx>> for NormalizationChecker<'tcx> {
    type Result = ControlFlow<()>;

    fn visit_ty(&mut self, t: Ty<'tcx>) -> Self::Result {
        match self.instance.try_instantiate_mir_and_normalize_erasing_regions(
            self.tcx,
            ty::TypingEnv::fully_monomorphized(),
            ty::EarlyBinder::bind(t),
        ) {
            Ok(_) => ControlFlow::Continue(()),
            Err(_) => ControlFlow::Break(()),
        }
    }
}

pub(crate) fn check_normalization_error<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
    body: &Body<'tcx>,
) {
    let mut checker = NormalizationChecker { tcx, instance };
    if body.visit_with(&mut checker).is_break() {
        // Plenty of code paths later assume that everything can be normalized.
        // Check normalization here to provide better diagnostics.
        // Normalization errors here are usually due to trait solving overflow.
        // FIXME: I assume that there are few type errors at post-analysis stage, but not
        // entirely sure.
        let def_id = instance.def_id();
        let def_span = tcx.def_span(def_id);
        let def_path_str = tcx.def_path_str(def_id);
        tcx.dcx().emit_fatal(RecursionLimit { span: def_span, instance, def_span, def_path_str });
    }
}
