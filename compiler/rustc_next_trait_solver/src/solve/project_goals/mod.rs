mod anon_const;
mod free_alias;
mod inherent;
mod opaque_types;

use rustc_type_ir::search_graph::IncreaseDepthForNested;
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

                // In the new solver, nested goals are evaluated eagerly and the ambiguous nested
                // goals aren't propagated back to the caller but discarded.
                //
                // This behavior made some unfortunate regressions when proving some associated
                // projection goal, as we may lose the nested goals from the impl candidate's
                // header which were preserved in the old solver. (See #122687)
                //
                // To avoid this, we create a `NormalizesTo` whose expected term is fully
                // unconstrained and evaluate it as if we are calling a function rather than
                // evaluating a nested goal, and register its ambiguous nested goals to the
                // caller's context instead of discarding them.
                //
                // This may feel incorrect as we don't reevaluate the ambiguous `NormalizesTo` goal
                // like other ordinary nested goals. Will we be in trouble if equating
                // `unconstrained_term` with the expected term results in inference constraints in
                // the impl header, which should be result in `Certainty::Yes` if we reevaluate it?
                // Hopefully not, because we apply the certainty of nested `NormalizesTo` goal as
                // the shallow certainty of the outer associated projection goal. If `NormalizesTo`
                // returns ambiguity, the caller will do so as well and it and its `NormalizesTo`
                // will be evaluated again by the caller's outer context.
                let (
                    NestedNormalizationGoals(nested_goals),
                    GoalEvaluation { goal: _, certainty, stalled_on: _, has_changed: _ },
                ) = self.evaluate_goal_raw(
                    GoalSource::TypeRelating,
                    normalizes_to,
                    None,
                    // We don't increase depth for nested goals for this `NormalizesTo` goal, as
                    // evaluating `NormalizesTo` is an extra step only exists in the new solver
                    // that behaves like a function call rather than an independent nested goal
                    // evaluation, so increasing the depth may end up regressions which hit the
                    // recursion limits for crates compiled well with the old solver. Furthermore,
                    // those nested goals from `NormalizesTo` will be evaluated again as the
                    // caller's nested goals with increased depths anyway.
                    IncreaseDepthForNested::No,
                )?;

                trace!(?nested_goals);

                // FIXME: We shouldn't be doing this in the long term in favor of eager
                // normalization.
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

            ty::AliasTermKind::InherentTy { def_id } => {
                self.normalize_inherent_associated_term(goal, def_id.into())
            }
            ty::AliasTermKind::InherentConst { def_id } => {
                self.normalize_inherent_associated_term(goal, def_id.into())
            }
            ty::AliasTermKind::OpaqueTy { def_id } => self.normalize_opaque_type(goal, def_id),
            ty::AliasTermKind::FreeTy { .. } | ty::AliasTermKind::FreeConst { .. } => {
                self.normalize_free_alias(goal).map_err(Into::into)
            }
            ty::AliasTermKind::AnonConst { def_id } => {
                self.normalize_anon_const(goal, def_id).map_err(Into::into)
            }
        }
    }
}
