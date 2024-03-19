//! Data structure used to inspect trait solver behavior.
//!
//! During trait solving we optionally build "proof trees", the root of
//! which is a [GoalEvaluation] with [GoalEvaluationKind::Root]. These
//! trees are used to improve the debug experience and are also used by
//! the compiler itself to provide necessary context for error messages.
//!
//! Because each nested goal in the solver gets [canonicalized] separately
//! and we discard inference progress via "probes", we cannot mechanically
//! use proof trees without somehow "lifting up" data local to the current
//! `InferCtxt`. Any data used mechanically is therefore canonicalized and
//! stored as [CanonicalState]. As printing canonicalized data worsens the
//! debugging dumps, we do not simply canonicalize everything.
//!
//! This means proof trees contain inference variables and placeholders
//! local to a different `InferCtxt` which must not be used with the
//! current one.
//!
//! [canonicalized]: https://rustc-dev-guide.rust-lang.org/solve/canonicalization.html

use super::{
    CandidateSource, Canonical, CanonicalInput, Certainty, Goal, GoalSource, NoSolution,
    QueryInput, QueryResult,
};
use crate::{infer::canonical::CanonicalVarValues, ty};
use format::ProofTreeFormatter;
use std::fmt::{Debug, Write};

mod format;

/// Some `data` together with information about how they relate to the input
/// of the canonical query.
///
/// This is only ever used as [CanonicalState]. Any type information in proof
/// trees used mechanically has to be canonicalized as we otherwise leak
/// inference variables from a nested `InferCtxt`.
#[derive(Debug, Clone, Copy, Eq, PartialEq, TypeFoldable, TypeVisitable)]
pub struct State<'tcx, T> {
    pub var_values: CanonicalVarValues<'tcx>,
    pub data: T,
}

pub type CanonicalState<'tcx, T> = Canonical<'tcx, State<'tcx, T>>;

/// When evaluating the root goals we also store the
/// original values for the `CanonicalVarValues` of the
/// canonicalized goal. We use this to map any [CanonicalState]
/// from the local `InferCtxt` of the solver query to
/// the `InferCtxt` of the caller.
#[derive(Eq, PartialEq)]
pub enum GoalEvaluationKind<'tcx> {
    Root { orig_values: Vec<ty::GenericArg<'tcx>> },
    Nested,
}

#[derive(Eq, PartialEq)]
pub struct GoalEvaluation<'tcx> {
    pub uncanonicalized_goal: Goal<'tcx, ty::Predicate<'tcx>>,
    pub kind: GoalEvaluationKind<'tcx>,
    pub evaluation: CanonicalGoalEvaluation<'tcx>,
}

#[derive(Eq, PartialEq)]
pub struct CanonicalGoalEvaluation<'tcx> {
    pub goal: CanonicalInput<'tcx>,
    pub kind: CanonicalGoalEvaluationKind<'tcx>,
    pub result: QueryResult<'tcx>,
}

#[derive(Eq, PartialEq)]
pub enum CanonicalGoalEvaluationKind<'tcx> {
    Overflow,
    CycleInStack,
    ProvisionalCacheHit,
    Evaluation { revisions: &'tcx [GoalEvaluationStep<'tcx>] },
}
impl Debug for GoalEvaluation<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        ProofTreeFormatter::new(f).format_goal_evaluation(self)
    }
}

#[derive(Eq, PartialEq)]
pub struct AddedGoalsEvaluation<'tcx> {
    pub evaluations: Vec<Vec<GoalEvaluation<'tcx>>>,
    pub result: Result<Certainty, NoSolution>,
}

#[derive(Eq, PartialEq)]
pub struct GoalEvaluationStep<'tcx> {
    pub instantiated_goal: QueryInput<'tcx, ty::Predicate<'tcx>>,

    /// The actual evaluation of the goal, always `ProbeKind::Root`.
    pub evaluation: Probe<'tcx>,
}

/// A self-contained computation during trait solving. This either
/// corresponds to a `EvalCtxt::probe(_X)` call or the root evaluation
/// of a goal.
#[derive(Eq, PartialEq)]
pub struct Probe<'tcx> {
    /// What happened inside of this probe in chronological order.
    pub steps: Vec<ProbeStep<'tcx>>,
    pub kind: ProbeKind<'tcx>,
}

impl Debug for Probe<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        ProofTreeFormatter::new(f).format_probe(self)
    }
}

#[derive(Eq, PartialEq)]
pub enum ProbeStep<'tcx> {
    /// We added a goal to the `EvalCtxt` which will get proven
    /// the next time `EvalCtxt::try_evaluate_added_goals` is called.
    AddGoal(GoalSource, CanonicalState<'tcx, Goal<'tcx, ty::Predicate<'tcx>>>),
    /// The inside of a `EvalCtxt::try_evaluate_added_goals` call.
    EvaluateGoals(AddedGoalsEvaluation<'tcx>),
    /// A call to `probe` while proving the current goal. This is
    /// used whenever there are multiple candidates to prove the
    /// current goalby .
    NestedProbe(Probe<'tcx>),
    CommitIfOkStart,
    CommitIfOkSuccess,
}

/// What kind of probe we're in. In case the probe represents a candidate, or
/// the final result of the current goal - via [ProbeKind::Root] - we also
/// store the [QueryResult].
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum ProbeKind<'tcx> {
    /// The root inference context while proving a goal.
    Root { result: QueryResult<'tcx> },
    /// Probe entered when normalizing the self ty during candidate assembly
    NormalizedSelfTyAssembly,
    /// Some candidate to prove the current goal.
    ///
    /// FIXME: Remove this in favor of always using more strongly typed variants.
    MiscCandidate { name: &'static str, result: QueryResult<'tcx> },
    /// A candidate for proving a trait or alias-relate goal.
    TraitCandidate { source: CandidateSource, result: QueryResult<'tcx> },
    /// Used in the probe that wraps normalizing the non-self type for the unsize
    /// trait, which is also structurally matched on.
    UnsizeAssembly,
    /// A call to `EvalCtxt::commit_if_ok` which failed, causing the work
    /// to be discarded.
    CommitIfOk,
    /// During upcasting from some source object to target object type, used to
    /// do a probe to find out what projection type(s) may be used to prove that
    /// the source type upholds all of the target type's object bounds.
    UpcastProjectionCompatibility,
}
