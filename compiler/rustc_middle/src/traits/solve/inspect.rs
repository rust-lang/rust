use super::{CanonicalInput, Certainty, Goal, NoSolution, QueryInput, QueryResult};
use crate::{traits::IsNormalizesToHack, ty};
use std::fmt::{Debug, Write};

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

struct ProofTreeFormatter<'a, 'b> {
    f: &'a mut (dyn Write + 'b),
    on_newline: bool,
}

impl Write for ProofTreeFormatter<'_, '_> {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        for line in s.split_inclusive("\n") {
            if self.on_newline {
                self.f.write_str("    ")?;
            }
            self.on_newline = line.ends_with("\n");
            self.f.write_str(line)?;
        }

        Ok(())
    }
}

impl ProofTreeFormatter<'_, '_> {
    fn nested(&mut self) -> ProofTreeFormatter<'_, '_> {
        ProofTreeFormatter { f: self, on_newline: true }
    }

    fn format_goal_evaluation(&mut self, goal: &GoalEvaluation<'_>) -> std::fmt::Result {
        let f = &mut *self.f;

        let goal_text = match goal.is_normalizes_to_hack {
            IsNormalizesToHack::Yes => "NORMALIZES-TO HACK GOAL",
            IsNormalizesToHack::No => "GOAL",
        };

        writeln!(f, "{}: {:?}", goal_text, goal.uncanonicalized_goal,)?;
        writeln!(f, "CANONICALIZED: {:?}", goal.canonicalized_goal)?;

        match &goal.kind {
            GoalEvaluationKind::CacheHit(CacheHit::Global) => {
                writeln!(f, "GLOBAL CACHE HIT: {:?}", goal.result)
            }
            GoalEvaluationKind::CacheHit(CacheHit::Provisional) => {
                writeln!(f, "PROVISIONAL CACHE HIT: {:?}", goal.result)
            }
            GoalEvaluationKind::Uncached { revisions } => {
                for (n, step) in revisions.iter().enumerate() {
                    let f = &mut *self.f;
                    writeln!(f, "REVISION {n}: {:?}", step.result)?;
                    let mut f = self.nested();
                    f.format_evaluation_step(step)?;
                }

                let f = &mut *self.f;
                writeln!(f, "RESULT: {:?}", goal.result)
            }
        }?;

        if goal.returned_goals.len() > 0 {
            let f = &mut *self.f;
            writeln!(f, "NESTED GOALS ADDED TO CALLER: [")?;
            let mut f = self.nested();
            for goal in goal.returned_goals.iter() {
                writeln!(f, "ADDED GOAL: {:?},", goal)?;
            }
            writeln!(self.f, "]")?;
        }

        Ok(())
    }

    fn format_evaluation_step(
        &mut self,
        evaluation_step: &GoalEvaluationStep<'_>,
    ) -> std::fmt::Result {
        let f = &mut *self.f;
        writeln!(f, "INSTANTIATED: {:?}", evaluation_step.instantiated_goal)?;

        for candidate in &evaluation_step.candidates {
            let mut f = self.nested();
            f.format_candidate(candidate)?;
        }
        for nested_goal_evaluation in &evaluation_step.nested_goal_evaluations {
            let mut f = self.nested();
            f.format_nested_goal_evaluation(nested_goal_evaluation)?;
        }

        Ok(())
    }

    fn format_candidate(&mut self, candidate: &GoalCandidate<'_>) -> std::fmt::Result {
        let f = &mut *self.f;

        match &candidate.kind {
            CandidateKind::NormalizedSelfTyAssembly => {
                writeln!(f, "NORMALIZING SELF TY FOR ASSEMBLY:")
            }
            CandidateKind::Candidate { name, result } => {
                writeln!(f, "CANDIDATE {}: {:?}", name, result)
            }
        }?;

        let mut f = self.nested();
        for candidate in &candidate.candidates {
            f.format_candidate(candidate)?;
        }
        for nested_evaluations in &candidate.nested_goal_evaluations {
            f.format_nested_goal_evaluation(nested_evaluations)?;
        }

        Ok(())
    }

    fn format_nested_goal_evaluation(
        &mut self,
        nested_goal_evaluation: &AddedGoalsEvaluation<'_>,
    ) -> std::fmt::Result {
        let f = &mut *self.f;
        writeln!(f, "TRY_EVALUATE_ADDED_GOALS: {:?}", nested_goal_evaluation.result)?;

        for (n, revision) in nested_goal_evaluation.evaluations.iter().enumerate() {
            let f = &mut *self.f;
            writeln!(f, "REVISION {n}")?;
            let mut f = self.nested();
            for goal_evaluation in revision {
                f.format_goal_evaluation(goal_evaluation)?;
            }
        }

        Ok(())
    }
}
