use rustc_data_structures::fx::FxIndexSet;
use rustc_infer::{infer::InferCtxt, traits::PredicateObligation};
use rustc_middle::traits::query::NoSolution;
use rustc_middle::traits::solve::{Certainty, MaybeCause};
use rustc_middle::traits::solve::inspect::RootGoalEvaluation;

use crate::solve::{InferCtxtEvalExt, GenerateProofTree, UseGlobalCache};
use crate::traits::IntercrateAmbiguityCause;

pub(crate) fn compute_intercrate_ambiguity_causes<'tcx>(
    infcx: &InferCtxt<'tcx>,
    obligations: &[PredicateObligation<'tcx>],
) -> FxIndexSet<IntercrateAmbiguityCause> {
    let mut causes: FxIndexSet<IntercrateAmbiguityCause> = Default::default();

    for obligation in obligations {
        infcx.probe(|_| {
            let (result, proof_tree) = infcx.evaluate_root_goal(obligation.clone().into(), GenerateProofTree::Yes(UseGlobalCache::No));
            let proof_tree = proof_tree.unwrap();
            match result {
                Ok((_has_changed, certainty, _nested)) => {
                    if certainty == Certainty::Maybe(MaybeCause::Ambiguity) {
                        search_ambiguity_causes(proof_tree, &mut causes);
                    }
                },
                Err(NoSolution) => unreachable!(),
            }
        })       
    }

    causes
}

fn search_ambiguity_causes<'tcx>(proof_tree: RootGoalEvaluation<'tcx>, causes: &mut FxIndexSet<IntercrateAmbiguityCause>) {
    // TODO
}