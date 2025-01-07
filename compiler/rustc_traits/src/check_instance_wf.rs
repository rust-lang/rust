use rustc_infer::infer::TyCtxtInferExt as _;
use rustc_middle::ty::{self, TyCtxt, TypingEnv};
use rustc_trait_selection::error_reporting::InferCtxtErrorExt;
use rustc_trait_selection::traits::{Obligation, ObligationCause, ObligationCtxt};

pub(crate) fn check_instance_wf<'tcx>(tcx: TyCtxt<'tcx>, instance: ty::Instance<'tcx>) {
    let ty::InstanceKind::Item(def_id) = instance.def else {
        // Probably want other instances too...:
        return;
    };

    let (infcx, param_env) =
        tcx.infer_ctxt().build_with_typing_env(TypingEnv::fully_monomorphized());
    let ocx = ObligationCtxt::new_with_diagnostics(&infcx);

    ocx.register_obligations(
        tcx.predicates_of(def_id).instantiate(tcx, instance.args).into_iter().map(
            |(clause, span)| {
                Obligation::new(tcx, ObligationCause::dummy_with_span(span), param_env, clause)
            },
        ),
    );

    let errors = ocx.select_all_or_error();
    if !errors.is_empty() {
        infcx.err_ctxt().report_fulfillment_errors(errors);
    }
}
