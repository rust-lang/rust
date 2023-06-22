use super::{
    CanonicalInput, Certainty, Goal, IsNormalizesToHack, NoSolution, QueryInput, QueryResult,
};
use crate::ty;
use format::ProofTreeFormatter;
use std::fmt::{Debug, Write};

mod format;

#[derive(Eq, PartialEq, Debug, Hash, HashStable)]
pub enum CacheHit {
    Provisional,
    Global,
}

#[derive(Eq, PartialEq, Hash, HashStable)]
pub struct GoalEvaluation<'tcx> {
    pub uncanonicalized_goal: Goal<'tcx, ty::Predicate<'tcx>>,
    pub canonicalized_goal: CanonicalInput<'tcx>,

    pub kind: GoalEvaluationKind<'tcx>,
    pub is_normalizes_to_hack: IsNormalizesToHack,
    pub returned_goals: Vec<Goal<'tcx, ty::Predicate<'tcx>>>,

    pub result: QueryResult<'tcx>,
}

#[derive(Eq, PartialEq, Hash, HashStable)]
pub enum GoalEvaluationKind<'tcx> {
    CacheHit(CacheHit),
    Uncached { revisions: Vec<GoalEvaluationStep<'tcx>> },
}
impl Debug for GoalEvaluation<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        ProofTreeFormatter { f, on_newline: true }.format_goal_evaluation(self)
    }
}

#[derive(Eq, PartialEq, Hash, HashStable)]
pub struct AddedGoalsEvaluation<'tcx> {
    pub evaluations: Vec<Vec<GoalEvaluation<'tcx>>>,
    pub result: Result<Certainty, NoSolution>,
}
impl Debug for AddedGoalsEvaluation<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        ProofTreeFormatter { f, on_newline: true }.format_nested_goal_evaluation(self)
    }
}

#[derive(Eq, PartialEq, Hash, HashStable)]
pub struct GoalEvaluationStep<'tcx> {
    pub instantiated_goal: QueryInput<'tcx, ty::Predicate<'tcx>>,

    pub nested_goal_evaluations: Vec<AddedGoalsEvaluation<'tcx>>,
    pub candidates: Vec<GoalCandidate<'tcx>>,

    pub result: QueryResult<'tcx>,
}
impl Debug for GoalEvaluationStep<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        ProofTreeFormatter { f, on_newline: true }.format_evaluation_step(self)
    }
}

#[derive(Eq, PartialEq, Hash, HashStable)]
pub struct GoalCandidate<'tcx> {
    pub nested_goal_evaluations: Vec<AddedGoalsEvaluation<'tcx>>,
    pub candidates: Vec<GoalCandidate<'tcx>>,
    pub kind: CandidateKind<'tcx>,
}

#[derive(Eq, PartialEq, Debug, Hash, HashStable)]
pub enum CandidateKind<'tcx> {
    /// Probe entered when normalizing the self ty during candidate assembly
    NormalizedSelfTyAssembly,
    /// A normal candidate for proving a goal
    Candidate { name: String, result: QueryResult<'tcx> },
}
impl Debug for GoalCandidate<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        ProofTreeFormatter { f, on_newline: true }.format_candidate(self)
    }
}
