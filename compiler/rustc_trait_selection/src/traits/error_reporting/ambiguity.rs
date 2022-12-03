use rustc_hir::def_id::DefId;
use rustc_infer::infer::InferCtxt;
use rustc_infer::traits::{Obligation, ObligationCause, TraitObligation};
use rustc_span::DUMMY_SP;

use crate::traits::ObligationCtxt;

pub fn recompute_applicable_impls<'tcx>(
    infcx: &InferCtxt<'tcx>,
    obligation: &TraitObligation<'tcx>,
) -> Vec<DefId> {
    let tcx = infcx.tcx;
    let param_env = obligation.param_env;
    let dummy_cause = ObligationCause::dummy();
    let impl_may_apply = |impl_def_id| {
        let ocx = ObligationCtxt::new_in_snapshot(infcx);
        let placeholder_obligation =
            infcx.replace_bound_vars_with_placeholders(obligation.predicate);
        let obligation_trait_ref =
            ocx.normalize(&dummy_cause, param_env, placeholder_obligation.trait_ref);

        let impl_substs = infcx.fresh_substs_for_item(DUMMY_SP, impl_def_id);
        let impl_trait_ref = tcx.bound_impl_trait_ref(impl_def_id).unwrap().subst(tcx, impl_substs);
        let impl_trait_ref = ocx.normalize(&ObligationCause::dummy(), param_env, impl_trait_ref);

        if let Err(_) = ocx.eq(&dummy_cause, param_env, obligation_trait_ref, impl_trait_ref) {
            return false;
        }

        let impl_predicates = tcx.predicates_of(impl_def_id).instantiate(tcx, impl_substs);
        ocx.register_obligations(
            impl_predicates
                .predicates
                .iter()
                .map(|&predicate| Obligation::new(tcx, dummy_cause.clone(), param_env, predicate)),
        );

        ocx.select_where_possible().is_empty()
    };

    let mut impls = Vec::new();
    tcx.for_each_relevant_impl(
        obligation.predicate.def_id(),
        obligation.predicate.skip_binder().trait_ref.self_ty(),
        |impl_def_id| {
            if infcx.probe(move |_snapshot| impl_may_apply(impl_def_id)) {
                impls.push(impl_def_id)
            }
        },
    );
    impls
}
