mod anon_const;
mod free_alias;
mod inherent;
mod opaque_types;

use rustc_type_ir::solve::QueryResultOrRerunNonErased;
use rustc_type_ir::{self as ty, Interner, ProjectionPredicate};
use tracing::{instrument, trace};

use crate::delegate::SolverDelegate;
use crate::solve::{
    Certainty, EvalCtxt, Goal, GoalEvaluation, GoalSource, NestedNormalizationGoals,
};

impl<D, I> EvalCtxt<'_, D>
where
    D: SolverDelegate<Interner = I>,
    I: Interner,
{
    #[instrument(level = "trace", skip(self), ret)]
    pub(super) fn compute_projection_goal(
        &mut self,
        goal: Goal<I, ProjectionPredicate<I>>,
    ) -> QueryResultOrRerunNonErased<I> {
        match goal.predicate.projection_term.kind {
            ty::AliasTermKind::ProjectionTy { .. } | ty::AliasTermKind::ProjectionConst { .. } => {
                let ty::ProjectionPredicate { projection_term: alias, term } = goal.predicate;
                let unconstrained_term = self.next_term_infer_of_kind(term);
                let normalizes_to =
                    goal.with(self.cx(), ty::NormalizesTo { alias, term: unconstrained_term });

                let (
                    NestedNormalizationGoals(nested_goals),
                    GoalEvaluation { goal: _, certainty, stalled_on: _, has_changed: _ },
                ) = self.evaluate_goal_raw(GoalSource::TypeRelating, normalizes_to, None)?;

                trace!(?nested_goals);

                // Normalize alias types in rhs. This is done in `EvalCtxt::add_goal` for nested
                // goals, but we might be evaluating the root goal.
                let term =
                    self.replace_alias_with_infer(term, GoalSource::TypeRelating, goal.param_env);

                // Add a `make_canonical_response` probe step so that we treat this as
                // a candidate, even if `try_evaluate_added_goals` bails due to an error.
                // It's `Certainty::AMBIGUOUS` because this candidate is not "finished",
                // since equating the normalized terms will lead to additional constraints.
                self.inspect.make_canonical_response(Certainty::AMBIGUOUS);

                // Apply the constraints.
                self.try_evaluate_added_goals()?;

                // Finally, equate the goal's RHS with the unconstrained var.
                //
                // SUBTLE:
                // We structurally relate aliases here. This is necessary
                // as we otherwise emit a nested `AliasRelate` goal in case the
                // returned term is a rigid alias, resulting in overflow.
                //
                // It is correct as both `goal.predicate.term` and `unconstrained_rhs`
                // start out as an unconstrained inference variable so any aliases get
                // fully normalized when instantiating it.
                //
                // FIXME: Strictly speaking this may be incomplete if the normalized-to
                // type contains an ambiguous alias referencing bound regions. We should
                // consider changing this to only use "shallow structural equality".
                self.eq_structurally_relating_aliases(goal.param_env, term, unconstrained_term)?;

                // Add the nested goals from normalization to our own nested goals.
                for (s, g) in nested_goals {
                    self.add_goal(s, g);
                }

                self.evaluate_added_goals_and_make_canonical_response(certainty)
            }

            ty::AliasTermKind::InherentTy { .. } => self.normalize_inherent_associated_term(goal),
            ty::AliasTermKind::InherentConst { .. } => {
                self.normalize_inherent_associated_term(goal)
            }
            ty::AliasTermKind::OpaqueTy { .. } => self.normalize_opaque_type(goal),
            ty::AliasTermKind::FreeTy { .. } | ty::AliasTermKind::FreeConst { .. } => {
                self.normalize_free_alias(goal).map_err(Into::into)
            }
            ty::AliasTermKind::AnonConst { .. } => {
                self.normalize_anon_const(goal).map_err(Into::into)
            }
        }
    }
}
