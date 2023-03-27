use rustc_hir::def_id::DefId;
use rustc_infer::infer::{InferCtxt, LateBoundRegionConversionTime};
use rustc_infer::traits::util::elaborate_predicates_with_span;
use rustc_infer::traits::{Obligation, ObligationCause, TraitObligation};
use rustc_middle::ty;
use rustc_span::{Span, DUMMY_SP};

use crate::traits::ObligationCtxt;

pub enum Ambiguity {
    DefId(DefId),
    ParamEnv(Span),
}

pub fn recompute_applicable_impls<'tcx>(
    infcx: &InferCtxt<'tcx>,
    obligation: &TraitObligation<'tcx>,
) -> Vec<Ambiguity> {
    let tcx = infcx.tcx;
    let param_env = obligation.param_env;

    let impl_may_apply = |impl_def_id| {
        let ocx = ObligationCtxt::new_in_snapshot(infcx);
        let placeholder_obligation =
            infcx.instantiate_binder_with_placeholders(obligation.predicate);
        let obligation_trait_ref =
            ocx.normalize(&ObligationCause::dummy(), param_env, placeholder_obligation.trait_ref);

        let impl_substs = infcx.fresh_substs_for_item(DUMMY_SP, impl_def_id);
        let impl_trait_ref = tcx.impl_trait_ref(impl_def_id).unwrap().subst(tcx, impl_substs);
        let impl_trait_ref = ocx.normalize(&ObligationCause::dummy(), param_env, impl_trait_ref);

        if let Err(_) =
            ocx.eq(&ObligationCause::dummy(), param_env, obligation_trait_ref, impl_trait_ref)
        {
            return false;
        }

        let impl_predicates = tcx.predicates_of(impl_def_id).instantiate(tcx, impl_substs);
        ocx.register_obligations(impl_predicates.predicates.iter().map(|&predicate| {
            Obligation::new(tcx, ObligationCause::dummy(), param_env, predicate)
        }));

        ocx.select_where_possible().is_empty()
    };

    let param_env_candidate_may_apply = |poly_trait_predicate: ty::PolyTraitPredicate<'tcx>| {
        let ocx = ObligationCtxt::new_in_snapshot(infcx);
        let placeholder_obligation =
            infcx.instantiate_binder_with_placeholders(obligation.predicate);
        let obligation_trait_ref =
            ocx.normalize(&ObligationCause::dummy(), param_env, placeholder_obligation.trait_ref);

        let param_env_predicate = infcx.instantiate_binder_with_fresh_vars(
            DUMMY_SP,
            LateBoundRegionConversionTime::HigherRankedType,
            poly_trait_predicate,
        );
        let param_env_trait_ref =
            ocx.normalize(&ObligationCause::dummy(), param_env, param_env_predicate.trait_ref);

        if let Err(_) =
            ocx.eq(&ObligationCause::dummy(), param_env, obligation_trait_ref, param_env_trait_ref)
        {
            return false;
        }

        ocx.select_where_possible().is_empty()
    };

    let mut ambiguities = Vec::new();

    tcx.for_each_relevant_impl(
        obligation.predicate.def_id(),
        obligation.predicate.skip_binder().trait_ref.self_ty(),
        |impl_def_id| {
            if infcx.probe(|_| impl_may_apply(impl_def_id)) {
                ambiguities.push(Ambiguity::DefId(impl_def_id))
            }
        },
    );

    let predicates =
        tcx.predicates_of(obligation.cause.body_id.to_def_id()).instantiate_identity(tcx);
    for (pred, span) in elaborate_predicates_with_span(tcx, predicates.into_iter()) {
        let kind = pred.kind();
        if let ty::PredicateKind::Clause(ty::Clause::Trait(trait_pred)) = kind.skip_binder()
            && param_env_candidate_may_apply(kind.rebind(trait_pred))
        {
            if kind.rebind(trait_pred.trait_ref) == ty::TraitRef::identity(tcx, trait_pred.def_id()) {
                ambiguities.push(Ambiguity::ParamEnv(tcx.def_span(trait_pred.def_id())))
            } else {
                ambiguities.push(Ambiguity::ParamEnv(span))
            }
        }
    }

    ambiguities
}
