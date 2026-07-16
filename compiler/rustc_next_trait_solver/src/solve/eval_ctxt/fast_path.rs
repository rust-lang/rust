//! This file contains a number of standalone functions useful for taking _fast paths_ in the trait
//! solver. The exact place where we check for these fast paths changes, and matters a lot for
//! performance. Ideally we'd only check them in `evaluate_goal`, but when evaluating root goals
//! we can check them earlier and save some time creating an `EvalCtxt` in the first place.
//!
//! For debugging, fast paths can be disabled using `-Zdisable-fast-paths`.

use rustc_type_ir::inherent::*;
use rustc_type_ir::solve::{
    Certainty, ComputeGoalFastPathOutcome, Goal, GoalStalledOn, GoalStalledOnOpaques,
    SucceededInErased,
};
use rustc_type_ir::{InferCtxtLike, Interner};

use crate::delegate::SolverDelegate;
use crate::solve::eval_ctxt::{RerunDecision, should_rerun_after_erased_canonicalization};
use crate::solve::{GoalEvaluation, HasChanged};

#[derive(Debug, Clone, Copy)]
pub(super) enum RerunStalled {
    WontMakeProgress(Certainty),
    MayMakeProgress,
}

/// If we have run a goal before, and it was stalled, check that any of the goal's
/// args have changed. This is a cheap way to determine that if we were to rerun this goal now,
/// it will remain stalled since it'll canonicalize the same way and evaluation is pure.
/// Therefore, we can skip this rerun
#[inline]
pub(super) fn rerunning_stalled_goal_may_make_progress<D, I>(
    delegate: &D,
    stalled_on: Option<&GoalStalledOn<I>>,
) -> RerunStalled
where
    D: SolverDelegate<Interner = I>,
    I: Interner,
{
    use RerunStalled::*;

    // If fast paths are turned off, then we assume all goals can always make progress
    if delegate.disable_trait_solver_fast_paths() {
        return MayMakeProgress;
    }

    // If the goal isn't stalled, we should definitely run it.
    let Some(&GoalStalledOn { ref opaques, ref stalled_vars, ref sub_roots, stalled_certainty }) =
        stalled_on
    else {
        return MayMakeProgress;
    };

    // If any of the stalled goal's generic arguments changed,
    // rerunning might make progress so we should rerun.
    if stalled_vars.iter().any(|value| delegate.is_changed_arg(*value)) {
        return MayMakeProgress;
    }

    // If some inference took place in any of the sub roots,
    // rerunning might make progress so we should rerun.
    if sub_roots.iter().any(|&vid| delegate.sub_unification_table_root_var(vid) != vid) {
        return MayMakeProgress;
    }

    match opaques {
        GoalStalledOnOpaques::No => {}
        &GoalStalledOnOpaques::Yes {
            num_opaques_in_storage,
            ref previously_succeeded_in_erased,
        } => {
            // If any opaques changed in the opaque type storage,
            // rerunning might make progress so we should rerun.
            if delegate
                .opaque_types_storage_num_entries()
                .needs_reevaluation(num_opaques_in_storage)
            {
                // Unless this goal previously succeeded in erased mode.
                // If the stalled goal successfully evaluated while erasing opaque types,
                // and the current state of the opaque type storage is not different in a way that is
                // relevant, this stalled goal cannot make any progress and we set this variable to true.
                let mut previous_erased_run_is_still_valid = false;

                if let &SucceededInErased::Yes { accessed_opaques } = previously_succeeded_in_erased
                {
                    match should_rerun_after_erased_canonicalization(
                        accessed_opaques,
                        delegate.typing_mode_raw(),
                        &delegate.clone_opaque_types_lookup_table(),
                    ) {
                        RerunDecision::Yes => {}
                        RerunDecision::EagerlyPropagateToParent => {
                            unreachable!("we never retry stalled queries if the parent was erased")
                        }
                        RerunDecision::No => {
                            previous_erased_run_is_still_valid = true;
                        }
                    }
                }

                if !previous_erased_run_is_still_valid {
                    return MayMakeProgress;
                }
            }
        }
    }

    // Otherwise, we can be sure that this stalled goal cannot make any progress
    // and we can exit early.
    WontMakeProgress(stalled_certainty)
}

/// `compute_goal_fast_path` is complicated enough that outling helps, so it gets optimized
/// separately from the caller. `compute_goal_fast_path` is the inlined version,
/// and most call sites (when adding goals) use it. However, when entering the root
/// we also want to check the fast path, and there the outlining matters.
///
/// FIXME(perf) cold might not be worth it here, given that we shuffled some things around since it
/// mattered.
#[cold]
#[inline(never)]
pub(super) fn compute_goal_fast_path_cold<D, I>(
    delegate: &D,
    goal: Goal<I, I::Predicate>,
    origin_span: I::Span,
) -> Option<GoalEvaluation<I>>
where
    D: SolverDelegate<Interner = I>,
    I: Interner,
{
    compute_goal_fast_path(delegate, goal, origin_span)
}

/// This is a fast path optimization:
/// See the docs on [`ComputeGoalFastPathOutcome`]
pub fn compute_goal_fast_path<D, I>(
    delegate: &D,
    goal: Goal<I, I::Predicate>,
    origin_span: I::Span,
) -> Option<GoalEvaluation<I>>
where
    D: SolverDelegate<Interner = I>,
    I: Interner,
{
    if delegate.disable_trait_solver_fast_paths() {
        return None;
    }

    match delegate.compute_goal_fast_path(goal, origin_span) {
        ComputeGoalFastPathOutcome::NoFastPath => None,
        ComputeGoalFastPathOutcome::TriviallyHolds => Some(GoalEvaluation {
            goal,
            certainty: Certainty::Yes,
            has_changed: HasChanged::No,
            stalled_on: None,
        }),
        ComputeGoalFastPathOutcome::TriviallyStalled { stalled_on } => Some(GoalEvaluation {
            goal,
            certainty: Certainty::AMBIGUOUS,
            has_changed: HasChanged::No,
            stalled_on: Some(stalled_on),
        }),
    }
}
