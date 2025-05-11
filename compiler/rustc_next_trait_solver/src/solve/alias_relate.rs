//! Implements the `AliasRelate` goal, which is used when unifying aliases.
//! Doing this via a separate goal is called "deferred alias relation" and part
//! of our more general approach to "lazy normalization".
//!
//! This is done by first structurally normalizing both sides of the goal, ending
//! up in either a concrete type, rigid alias, or an infer variable.
//! These are related further according to the rules below:
//!
//! (1.) If we end up with two rigid aliases, then we relate them structurally.
//!
//! (2.) If we end up with an infer var and a rigid alias, then we instantiate
//! the infer var with the constructor of the alias and then recursively relate
//! the terms.
//!
//! (3.) Otherwise, if we end with two rigid (non-projection) or infer types,
//! relate them structurally.

use rustc_type_ir::inherent::*;
use rustc_type_ir::solve::GoalSource;
use rustc_type_ir::{self as ty, Interner};
use tracing::{instrument, trace};

use crate::delegate::SolverDelegate;
use crate::solve::{Certainty, EvalCtxt, Goal, QueryResult};

impl<D, I> EvalCtxt<'_, D>
where
    D: SolverDelegate<Interner = I>,
    I: Interner,
{
    #[instrument(level = "trace", skip(self), ret)]
    pub(super) fn compute_alias_relate_goal(
        &mut self,
        goal: Goal<I, (I::Term, I::Term, ty::AliasRelationDirection)>,
    ) -> QueryResult<I> {
        let cx = self.cx();
        let Goal { param_env, predicate: (lhs, rhs, direction) } = goal;

        // Check that the alias-relate goal is reasonable. Writeback for
        // `coroutine_stalled_predicates` can replace alias terms with
        // `{type error}` if the alias still contains infer vars, so we also
        // accept alias-relate goals where one of the terms is an error.
        debug_assert!(
            lhs.to_alias_term().is_some()
                || rhs.to_alias_term().is_some()
                || lhs.is_error()
                || rhs.is_error()
        );

        // Structurally normalize the lhs.
        let lhs = if let Some(alias) = lhs.to_alias_term() {
            let term = self.next_term_infer_of_kind(lhs);
            self.add_goal(
                GoalSource::TypeRelating,
                goal.with(cx, ty::NormalizesTo { alias, term }),
            );
            term
        } else {
            lhs
        };

        // Structurally normalize the rhs.
        let rhs = if let Some(alias) = rhs.to_alias_term() {
            let term = self.next_term_infer_of_kind(rhs);
            self.add_goal(
                GoalSource::TypeRelating,
                goal.with(cx, ty::NormalizesTo { alias, term }),
            );
            term
        } else {
            rhs
        };

        // Add a `make_canonical_response` probe step so that we treat this as
        // a candidate, even if `try_evaluate_added_goals` bails due to an error.
        // It's `Certainty::AMBIGUOUS` because this candidate is not "finished",
        // since equating the normalized terms will lead to additional constraints.
        self.inspect.make_canonical_response(Certainty::AMBIGUOUS);

        // Apply the constraints.
        self.try_evaluate_added_goals()?;
        let lhs = self.resolve_vars_if_possible(lhs);
        let rhs = self.resolve_vars_if_possible(rhs);
        trace!(?lhs, ?rhs);

        let variance = match direction {
            ty::AliasRelationDirection::Equate => ty::Invariant,
            ty::AliasRelationDirection::Subtype => ty::Covariant,
        };
        match (lhs.to_alias_term(), rhs.to_alias_term()) {
            (None, None) => {
                self.relate(param_env, lhs, variance, rhs)?;
                self.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
            }

            (Some(alias), None) => {
                self.relate_rigid_alias_non_alias(param_env, alias, variance, rhs)?;
                self.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
            }
            (None, Some(alias)) => {
                self.relate_rigid_alias_non_alias(
                    param_env,
                    alias,
                    variance.xform(ty::Contravariant),
                    lhs,
                )?;
                self.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
            }

            (Some(alias_lhs), Some(alias_rhs)) => {
                self.relate(param_env, alias_lhs, variance, alias_rhs)?;
                self.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
            }
        }
    }
}
