pub use rustc_next_trait_solver::solve::*;

mod delegate;
pub mod inspect;
mod normalize;
mod rustc_fulfill;
mod select;

pub(crate) use delegate::SolverDelegate;
pub(crate) use normalize::deeply_normalize_for_diagnostics;
pub use normalize::{
    deeply_normalize, deeply_normalize_with_skipped_universes,
    deeply_normalize_with_skipped_universes_and_ambiguous_coroutine_goals, normalize,
};
pub use rustc_fulfill::{FulfillmentCtxt, NextSolverError};
use rustc_middle::query::Providers;
use rustc_middle::ty::TyCtxt;
pub use select::InferCtxtSelectExt;

fn evaluate_root_goal_for_proof_tree_raw<'tcx>(
    tcx: TyCtxt<'tcx>,
    key: (CanonicalInput<TyCtxt<'tcx>>, usize),
) -> (QueryResult<TyCtxt<'tcx>>, &'tcx inspect::Probe<TyCtxt<'tcx>>) {
    evaluate_root_goal_for_proof_tree_raw_provider::<SolverDelegate<'tcx>, TyCtxt<'tcx>>(
        tcx, key.0, key.1,
    )
}

pub fn provide(providers: &mut Providers) {
    *providers = Providers { evaluate_root_goal_for_proof_tree_raw, ..*providers };
}
