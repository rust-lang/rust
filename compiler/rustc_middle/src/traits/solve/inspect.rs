use super::{CandidateSource, CanonicalInput, Certainty, Goal, NoSolution, QueryResult};
use crate::infer::canonical::{Canonical, CanonicalVarValues};
use crate::ty;
use format::ProofTreeFormatter;
use std::fmt::{Debug, Write};

mod format;

#[derive(Debug, Clone, Eq, PartialEq, Hash, HashStable, TypeFoldable, TypeVisitable)]
pub struct State<'tcx, T> {
    pub var_values: CanonicalVarValues<'tcx>,
    pub data: T,
}
pub type CanonicalState<'tcx, T> = Canonical<'tcx, State<'tcx, T>>;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum CacheHit {
    Provisional,
    Global,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum IsNormalizesToHack {
    Yes,
    No,
}

pub struct RootGoalEvaluation<'tcx> {
    pub goal: Goal<'tcx, ty::Predicate<'tcx>>,
    pub orig_values: Vec<ty::GenericArg<'tcx>>,
    pub evaluation: CanonicalGoalEvaluation<'tcx>,
    pub returned_goals: Vec<Goal<'tcx, ty::Predicate<'tcx>>>,
}

pub struct NestedGoalEvaluation<'tcx> {
    pub goal: CanonicalState<'tcx, Goal<'tcx, ty::Predicate<'tcx>>>,
    pub orig_values: CanonicalState<'tcx, Vec<ty::GenericArg<'tcx>>>,
    pub is_normalizes_to_hack: IsNormalizesToHack,
    pub evaluation: CanonicalGoalEvaluation<'tcx>,
    pub returned_goals: Vec<CanonicalState<'tcx, Goal<'tcx, ty::Predicate<'tcx>>>>,
}

pub struct CanonicalGoalEvaluation<'tcx> {
    pub goal: CanonicalInput<'tcx>,
    pub data: GoalEvaluationData<'tcx>,
    pub result: QueryResult<'tcx>,
}

pub enum GoalEvaluationData<'tcx> {
    CacheHit(CacheHit),
    Uncached { revisions: Vec<GoalEvaluationStep<'tcx>> },
}
impl Debug for RootGoalEvaluation<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        ProofTreeFormatter::new(f).format_root_goal_evaluation(self)
    }
}

pub struct AddedGoalsEvaluation<'tcx> {
    pub evaluations: Vec<Vec<NestedGoalEvaluation<'tcx>>>,
    pub result: Result<Certainty, NoSolution>,
}

pub struct GoalEvaluationStep<'tcx> {
    pub added_goals_evaluations: Vec<AddedGoalsEvaluation<'tcx>>,
    pub candidates: Vec<GoalCandidate<'tcx>>,

    pub result: QueryResult<'tcx>,
}

pub struct GoalCandidate<'tcx> {
    pub added_goals_evaluations: Vec<AddedGoalsEvaluation<'tcx>>,
    pub candidates: Vec<GoalCandidate<'tcx>>,
    pub kind: ProbeKind<'tcx>,
}

#[derive(Debug, PartialEq, Eq)]
pub enum ProbeKind<'tcx> {
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
