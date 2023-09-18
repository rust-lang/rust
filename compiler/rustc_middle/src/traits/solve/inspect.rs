use super::{
    CandidateSource, CanonicalInput, Certainty, Goal, IsNormalizesToHack, NoSolution, QueryInput,
    QueryResult,
};
use crate::ty;
use format::ProofTreeFormatter;
use std::fmt::{Debug, Write};

mod format;

#[derive(Debug, Eq, PartialEq)]
pub enum CacheHit {
    Provisional,
    Global,
}

#[derive(Eq, PartialEq)]
pub enum GoalEvaluationKind {
    Root,
    Nested { is_normalizes_to_hack: IsNormalizesToHack },
}

#[derive(Eq, PartialEq)]
pub struct GoalEvaluation<'tcx> {
    pub uncanonicalized_goal: Goal<'tcx, ty::Predicate<'tcx>>,
    pub kind: GoalEvaluationKind,
    pub evaluation: CanonicalGoalEvaluation<'tcx>,
    pub returned_goals: Vec<Goal<'tcx, ty::Predicate<'tcx>>>,
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
    CacheHit(CacheHit),
    Uncached { revisions: Vec<GoalEvaluationStep<'tcx>> },
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
    AddGoal(Goal<'tcx, ty::Predicate<'tcx>>),
    EvaluateGoals(AddedGoalsEvaluation<'tcx>),
    NestedProbe(Probe<'tcx>),
}

#[derive(Debug, PartialEq, Eq)]
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
    /// During upcasting from some source object to target object type, used to
    /// do a probe to find out what projection type(s) may be used to prove that
    /// the source type upholds all of the target type's object bounds.
    UpcastProjectionCompatibility,
}
