use rustc_hir::def_id::DefId;
use rustc_infer::infer::{BoundRegionConversionTime, InferCtxt};
use rustc_infer::traits::util::elaborate;
use rustc_infer::traits::{Obligation, ObligationCause, PolyTraitObligation};
use rustc_middle::ty;
use rustc_span::{Span, DUMMY_SP};

use crate::traits::query::evaluate_obligation::InferCtxtExt;
use crate::traits::ObligationCtxt;

#[derive(Debug)]
pub enum CandidateSource {
    DefId(DefId),
    ParamEnv(Span),
}

pub fn compute_applicable_impls_for_diagnostics<'tcx>(
    infcx: &InferCtxt<'tcx>,
    obligation: &PolyTraitObligation<'tcx>,
) -> Vec<CandidateSource> {
    let tcx = infcx.tcx;
    let param_env = obligation.param_env;

    let predicate_polarity = obligation.predicate.skip_binder().polarity;

    let impl_may_apply = |impl_def_id| {
        let ocx = ObligationCtxt::new(infcx);
        infcx.enter_forall(obligation.predicate, |placeholder_obligation| {
            let obligation_trait_ref = ocx.normalize(
                &ObligationCause::dummy(),
                param_env,
                placeholder_obligation.trait_ref,
            );

            let impl_args = infcx.fresh_args_for_item(DUMMY_SP, impl_def_id);
            let impl_trait_ref =
                tcx.impl_trait_ref(impl_def_id).unwrap().instantiate(tcx, impl_args);
            let impl_trait_ref =
                ocx.normalize(&ObligationCause::dummy(), param_env, impl_trait_ref);

            if let Err(_) =
                ocx.eq(&ObligationCause::dummy(), param_env, obligation_trait_ref, impl_trait_ref)
            {
                return false;
            }

            let impl_trait_header = tcx.impl_trait_header(impl_def_id).unwrap();
            let impl_polarity = impl_trait_header.polarity;

            match (impl_polarity, predicate_polarity) {
                (ty::ImplPolarity::Positive, ty::PredicatePolarity::Positive)
                | (ty::ImplPolarity::Negative, ty::PredicatePolarity::Negative) => {}
                _ => return false,
            }

            let obligations = tcx
                .predicates_of(impl_def_id)
                .instantiate(tcx, impl_args)
                .into_iter()
                .map(|(predicate, _)| {
                    Obligation::new(tcx, ObligationCause::dummy(), param_env, predicate)
                })
                // Kinda hacky, but let's just throw away obligations that overflow.
                // This may reduce the accuracy of this check (if the obligation guides
                // inference or it actually resulted in error after others are processed)
                // ... but this is diagnostics code.
                .filter(|obligation| {
                    infcx.next_trait_solver() || infcx.evaluate_obligation(obligation).is_ok()
                });
            ocx.register_obligations(obligations);

            ocx.select_where_possible().is_empty()
        })
    };

    let param_env_candidate_may_apply = |poly_trait_predicate: ty::PolyTraitPredicate<'tcx>| {
        let ocx = ObligationCtxt::new(infcx);
        infcx.enter_forall(obligation.predicate, |placeholder_obligation| {
            let obligation_trait_ref = ocx.normalize(
                &ObligationCause::dummy(),
                param_env,
                placeholder_obligation.trait_ref,
            );

            let param_env_predicate = infcx.instantiate_binder_with_fresh_vars(
                DUMMY_SP,
                BoundRegionConversionTime::HigherRankedType,
                poly_trait_predicate,
            );
            let param_env_trait_ref =
                ocx.normalize(&ObligationCause::dummy(), param_env, param_env_predicate.trait_ref);

            if let Err(_) = ocx.eq(
                &ObligationCause::dummy(),
                param_env,
                obligation_trait_ref,
                param_env_trait_ref,
            ) {
                return false;
            }

            ocx.select_where_possible().is_empty()
        })
    };

    let mut ambiguities = Vec::new();

    tcx.for_each_relevant_impl(
        obligation.predicate.def_id(),
        obligation.predicate.skip_binder().trait_ref.self_ty(),
        |impl_def_id| {
            if infcx.probe(|_| impl_may_apply(impl_def_id)) {
                ambiguities.push(CandidateSource::DefId(impl_def_id))
            }
        },
    );

    let predicates =
        tcx.predicates_of(obligation.cause.body_id.to_def_id()).instantiate_identity(tcx);
    for (pred, span) in elaborate(tcx, predicates.into_iter()) {
        let kind = pred.kind();
        if let ty::ClauseKind::Trait(trait_pred) = kind.skip_binder()
            && param_env_candidate_may_apply(kind.rebind(trait_pred))
        {
            if kind.rebind(trait_pred.trait_ref)
                == ty::Binder::dummy(ty::TraitRef::identity(tcx, trait_pred.def_id()))
            {
                ambiguities.push(CandidateSource::ParamEnv(tcx.def_span(trait_pred.def_id())))
            } else {
                ambiguities.push(CandidateSource::ParamEnv(span))
            }
        }
    }

    ambiguities
}
