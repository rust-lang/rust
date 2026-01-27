use std::mem;

use rustc_infer::infer::InferCtxt;
use rustc_infer::traits::{Obligation, ObligationCause};
use rustc_middle::hooks::TypeErasedInfcx;
pub use rustc_next_trait_solver::solve::*;

mod delegate;
mod fulfill;
pub mod inspect;
mod normalize;
mod select;

pub(crate) use delegate::SolverDelegate;
pub use fulfill::{FulfillmentCtxt, NextSolverError, StalledOnCoroutines};
pub(crate) use normalize::deeply_normalize_for_diagnostics;
pub use normalize::{
    deeply_normalize, deeply_normalize_with_skipped_universes,
    deeply_normalize_with_skipped_universes_and_ambiguous_coroutine_goals,
};
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_middle::util::Providers;
use rustc_span::Span;
pub use select::InferCtxtSelectExt;

use crate::traits::ObligationCtxt;

fn evaluate_root_goal_for_proof_tree_raw<'tcx>(
    tcx: TyCtxt<'tcx>,
    canonical_input: CanonicalInput<TyCtxt<'tcx>>,
) -> (QueryResult<TyCtxt<'tcx>>, &'tcx inspect::Probe<TyCtxt<'tcx>>) {
    evaluate_root_goal_for_proof_tree_raw_provider::<SolverDelegate<'tcx>, TyCtxt<'tcx>>(
        tcx,
        canonical_input,
    )
}

fn try_eagerly_normalize_alias<'a, 'tcx>(
    tcx: TyCtxt<'tcx>,
    type_erased_infcx: TypeErasedInfcx<'a, 'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    span: Span,
    alias: ty::AliasTy<'tcx>,
) -> Ty<'tcx> {
    let infcx = unsafe {
        mem::transmute::<TypeErasedInfcx<'a, 'tcx>, &'a InferCtxt<'tcx>>(type_erased_infcx)
    };

    let ocx = ObligationCtxt::new(infcx);

    let infer_term = infcx.next_ty_var(span);
    let obligation = Obligation::new(
        tcx,
        // we ignore the error anyway
        ObligationCause::dummy(),
        param_env,
        ty::PredicateKind::AliasRelate(
            alias.to_ty(tcx).into(),
            infer_term.into(),
            ty::AliasRelationDirection::Equate,
        ),
    );

    ocx.register_obligation(obligation);

    // This only tries to eagerly resolve, if it errors we don't care.
    let _ = ocx.try_evaluate_obligations();

    infcx.resolve_vars_if_possible(infer_term)
}

pub fn provide(providers: &mut Providers) {
    providers.hooks.try_eagerly_normalize_alias = try_eagerly_normalize_alias;
    providers.queries.evaluate_root_goal_for_proof_tree_raw = evaluate_root_goal_for_proof_tree_raw;
}
