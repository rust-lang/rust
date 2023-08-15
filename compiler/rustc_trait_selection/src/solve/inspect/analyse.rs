use std::ops::ControlFlow;

use rustc_data_structures::fx::FxIndexSet;
use rustc_infer::traits::TraitEngine;
use rustc_infer::{infer::InferCtxt, traits::PredicateObligation};
use rustc_middle::traits::query::NoSolution;
use rustc_middle::traits::solve::inspect::{self, RootGoalEvaluation};
use rustc_middle::traits::solve::{Certainty, Goal};
use rustc_middle::traits::ObligationCause;
use rustc_middle::ty;
use rustc_middle::ty::print::with_no_trimmed_paths;
use rustc_middle::ty::TypeVisitableExt;

use crate::solve::{GenerateProofTree, InferCtxtEvalExt, UseGlobalCache};
use crate::traits::coherence::{self, Conflict};
use crate::traits::TraitEngineExt;
use crate::traits::{IntercrateAmbiguityCause, StructurallyNormalizeExt};

pub struct InspectGoal<'a, 'tcx> {
    infcx: &'a InferCtxt<'tcx>,
    goal: Goal<'tcx, ty::Predicate<'tcx>>,
}

pub struct InspectCandidate<'a, 'tcx> {
    infcx: &'a InferCtxt<'tcx>,
}

impl<'a, 'tcx> InspectCandidate<'a, 'tcx> {
    pub fn infcx(&self) -> &'a InferCtxt<'tcx> {
        self.infcx
    }

    /// `None` in case this is the only candidate to prove the `goal`.
    pub fn kind(&self) -> Option<inspect::ProbeKind<'tcx>> {
        todo!()
    }

    pub fn result(&self) -> Result<Certainty, NoSolution> {
        todo!()
    }

    pub fn visit_nested<V: ProofTreeVisitor<'tcx>>(
        &self,
        visitor: &mut V,
    ) -> ControlFlow<V::BreakTy> {
        // TODO
        todo!()
    }
}

impl<'a, 'tcx> InspectGoal<'a, 'tcx> {
    pub fn infcx(&self) -> &'a InferCtxt<'tcx> {
        self.infcx
    }

    pub fn goal(&self) -> Goal<'tcx, ty::Predicate<'tcx>> {
        self.goal
    }

    pub fn result(&self) -> Result<Certainty, NoSolution> {
        // TODO
        todo!()
    }

    pub fn candidates(&self) -> Vec<InspectCandidate<'a, 'tcx>> {
        // TODO
        todo!()
    }

    fn new(infcx: &'a InferCtxt<'tcx>, root: &'a RootGoalEvaluation<'tcx>) -> Self {
        InspectGoal { infcx, goal: root.goal }
    }
}

/// The public API to interact with proof trees.
pub trait ProofTreeVisitor<'tcx> {
    type BreakTy;

    fn visit_goal(&mut self, goal: &InspectGoal<'_, 'tcx>) -> ControlFlow<Self::BreakTy>;
}

pub trait ProofTreeInferCtxtExt<'tcx> {
    fn visit_proof_tree<V: ProofTreeVisitor<'tcx>>(
        &self,
        root: &RootGoalEvaluation<'tcx>,
        visitor: &mut V,
    ) -> ControlFlow<V::BreakTy>;
}

impl<'tcx> ProofTreeInferCtxtExt<'tcx> for InferCtxt<'tcx> {
    fn visit_proof_tree<V: ProofTreeVisitor<'tcx>>(
        &self,
        root: &RootGoalEvaluation<'tcx>,
        visitor: &mut V,
    ) -> ControlFlow<V::BreakTy> {
        visitor.visit_goal(&InspectGoal::new(self, root))
    }
}

pub(crate) fn compute_intercrate_ambiguity_causes<'tcx>(
    infcx: &InferCtxt<'tcx>,
    obligations: &[PredicateObligation<'tcx>],
) -> FxIndexSet<IntercrateAmbiguityCause> {
    let mut causes: FxIndexSet<IntercrateAmbiguityCause> = Default::default();

    for obligation in obligations {
        infcx.probe(|_| {
            let (_, proof_tree) = infcx.evaluate_root_goal(
                obligation.clone().into(),
                GenerateProofTree::Yes(UseGlobalCache::No),
            );
            let proof_tree = proof_tree.unwrap();
            search_ambiguity_causes(infcx, proof_tree, &mut causes);
        })
    }

    causes
}

struct AmbiguityCausesVisitor<'a> {
    causes: &'a mut FxIndexSet<IntercrateAmbiguityCause>,
}

impl<'a, 'tcx> ProofTreeVisitor<'tcx> for AmbiguityCausesVisitor<'a> {
    type BreakTy = !;
    fn visit_goal(&mut self, goal: &InspectGoal<'_, 'tcx>) -> ControlFlow<Self::BreakTy> {
        let infcx = goal.infcx();
        for cand in goal.candidates() {
            cand.visit_nested(self)?;
        }
        // When searching for intercrate ambiguity causes, we only need to look
        // at ambiguous goals, as for others the coherence unknowable candidate
        // was irrelevant.
        match goal.result() {
            Ok(Certainty::Maybe(_)) => {}
            Ok(Certainty::Yes) | Err(NoSolution) => return ControlFlow::Continue(()),
        }

        let Goal { param_env, predicate } = goal.goal();

        let trait_ref = match predicate.kind().no_bound_vars() {
            Some(ty::PredicateKind::Clause(ty::ClauseKind::Trait(tr))) => tr.trait_ref,
            Some(ty::PredicateKind::Clause(ty::ClauseKind::Projection(proj))) => {
                proj.projection_ty.trait_ref(infcx.tcx)
            }
            _ => return ControlFlow::Continue(()),
        };

        let mut ambiguity_cause = None;
        for cand in goal.candidates() {
            match goal.result() {
                Ok(Certainty::Maybe(_)) => {}
                // We only add intercrate ambiguity causes if the goal would
                // otherwise result in an error.
                //
                // FIXME: this isn't quite right. Changing a goal from YES with
                // inference contraints to AMBIGUOUS can also cause a goal to not
                // fail.
                Ok(Certainty::Yes) => {
                    ambiguity_cause = None;
                    break;
                }
                Err(NoSolution) => continue,
            }

            // FIXME: boiiii, using string comparisions here sure is scuffed.
            if let Some(inspect::ProbeKind::MiscCandidate {
                name: "coherence unknowable",
                result: _,
            }) = cand.kind()
            {
                let lazily_normalize_ty = |ty| {
                    let mut fulfill_cx = <dyn TraitEngine<'tcx>>::new(infcx);
                    match infcx
                        .at(&ObligationCause::dummy(), param_env)
                        .structurally_normalize(ty, &mut *fulfill_cx)
                    {
                        Ok(ty) => Ok(ty),
                        Err(_errs) => Err(()),
                    }
                };

                infcx.probe(|_| {
                    match coherence::trait_ref_is_knowable(
                        infcx.tcx,
                        trait_ref,
                        lazily_normalize_ty,
                    ) {
                        Err(()) => {}
                        Ok(Ok(())) => warn!("expected an unknowable trait ref: {trait_ref:?}"),
                        Ok(Err(conflict)) => {
                            if !trait_ref.references_error() {
                                let self_ty = trait_ref.self_ty();
                                let (trait_desc, self_desc) = with_no_trimmed_paths!({
                                    let trait_desc = trait_ref.print_only_trait_path().to_string();
                                    let self_desc = self_ty
                                        .has_concrete_skeleton()
                                        .then(|| self_ty.to_string());
                                    (trait_desc, self_desc)
                                });
                                ambiguity_cause = Some(match conflict {
                                    Conflict::Upstream => {
                                        IntercrateAmbiguityCause::UpstreamCrateUpdate {
                                            trait_desc,
                                            self_desc,
                                        }
                                    }
                                    Conflict::Downstream => {
                                        IntercrateAmbiguityCause::DownstreamCrate {
                                            trait_desc,
                                            self_desc,
                                        }
                                    }
                                });
                            }
                        }
                    }
                })
            }
        }

        if let Some(ambiguity_cause) = ambiguity_cause {
            self.causes.insert(ambiguity_cause);
        }

        ControlFlow::Continue(())
    }
}

fn search_ambiguity_causes<'tcx>(
    infcx: &InferCtxt<'tcx>,
    proof_tree: inspect::RootGoalEvaluation<'tcx>,
    causes: &mut FxIndexSet<IntercrateAmbiguityCause>,
) {
    infcx.visit_proof_tree(&proof_tree, &mut AmbiguityCausesVisitor { causes });
}
