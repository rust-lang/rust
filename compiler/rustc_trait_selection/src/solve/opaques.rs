use rustc_middle::traits::query::NoSolution;
use rustc_middle::traits::solve::{Certainty, Goal, QueryResult};
use rustc_middle::traits::Reveal;
use rustc_middle::ty::{self, ProjectionPredicate};

use super::{EvalCtxt, SolverMode};

impl<'tcx> EvalCtxt<'_, 'tcx> {
    pub(super) fn normalize_opaque_type(
        &mut self,
        goal: Goal<'tcx, ProjectionPredicate<'tcx>>,
    ) -> QueryResult<'tcx> {
        let tcx = self.tcx();
        let opaque_ty = goal.predicate.projection_ty;
        let expected = goal.predicate.term.ty().unwrap();
        let mut candidates = Vec::new();

        candidates.extend(
            match goal.param_env.reveal() {
                // Failing to reveal opaque types has to result in
                // ambiguity during coherence.
                Reveal::UserFacing => match self.solver_mode() {
                    SolverMode::Normal => Err(NoSolution),
                    SolverMode::Coherence => {
                        self.evaluate_added_goals_and_make_canonical_response(Certainty::AMBIGUOUS)
                    }
                },
                Reveal::All => self.probe(|this| {
                    let actual = tcx.type_of(opaque_ty.def_id).subst(tcx, opaque_ty.substs);
                    this.eq(goal.param_env, expected, actual)?;
                    this.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
                }),
            }
            .ok(),
        );

        for assumption in goal.param_env.caller_bounds().iter() {
            match assumption.kind().skip_binder() {
                ty::PredicateKind::DefineOpaque(env_opaque_ty, actual) => {
                    let result = self.probe(|this| {
                        this.eq(goal.param_env, opaque_ty, env_opaque_ty)?;
                        this.eq(goal.param_env, expected, actual)?;
                        this.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
                    });
                    candidates.extend(result.ok());
                }

                ty::PredicateKind::Clause(_)
                | ty::PredicateKind::WellFormed(_)
                | ty::PredicateKind::ObjectSafe(_)
                | ty::PredicateKind::ClosureKind(_, _, _)
                | ty::PredicateKind::Subtype(_)
                | ty::PredicateKind::Coerce(_)
                | ty::PredicateKind::ConstEvaluatable(_)
                | ty::PredicateKind::ConstEquate(_, _)
                | ty::PredicateKind::TypeWellFormedFromEnv(_)
                | ty::PredicateKind::Ambiguous
                | ty::PredicateKind::AliasRelate(_, _, _) => {}
            }
        }

        if let Some(result) = self.try_merge_responses(&candidates) {
            Ok(result)
        } else {
            self.flounder(&candidates)
        }
    }
}
