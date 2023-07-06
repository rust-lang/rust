use super::*;

pub(super) struct ProofTreeFormatter<'a, 'b> {
    f: &'a mut (dyn Write + 'b),
}

/// A formatter which adds 4 spaces of indentation to its input before
/// passing it on to its nested formatter.
///
/// We can use this for arbitrary levels of indentation by nesting it.
struct Indentor<'a, 'b> {
    f: &'a mut (dyn Write + 'b),
    on_newline: bool,
}

impl Write for Indentor<'_, '_> {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        for line in s.split_inclusive('\n') {
            if self.on_newline {
                self.f.write_str("    ")?;
            }
            self.on_newline = line.ends_with('\n');
            self.f.write_str(line)?;
        }

        Ok(())
    }
}

impl<'a, 'b> ProofTreeFormatter<'a, 'b> {
    pub(super) fn new(f: &'a mut (dyn Write + 'b)) -> Self {
        ProofTreeFormatter { f }
    }

    fn nested<F, R>(&mut self, func: F) -> R
    where
        F: FnOnce(&mut ProofTreeFormatter<'_, '_>) -> R,
    {
        func(&mut ProofTreeFormatter { f: &mut Indentor { f: self.f, on_newline: true } })
    }

    pub(super) fn format_goal_evaluation(&mut self, goal: &GoalEvaluation<'_>) -> std::fmt::Result {
        let goal_text = match goal.is_normalizes_to_hack {
            IsNormalizesToHack::Yes => "NORMALIZES-TO HACK GOAL",
            IsNormalizesToHack::No => "GOAL",
        };

        writeln!(self.f, "{}: {:?}", goal_text, goal.uncanonicalized_goal)?;
        writeln!(self.f, "CANONICALIZED: {:?}", goal.canonicalized_goal)?;

        match &goal.kind {
            GoalEvaluationKind::CacheHit(CacheHit::Global) => {
                writeln!(self.f, "GLOBAL CACHE HIT: {:?}", goal.result)
            }
            GoalEvaluationKind::CacheHit(CacheHit::Provisional) => {
                writeln!(self.f, "PROVISIONAL CACHE HIT: {:?}", goal.result)
            }
            GoalEvaluationKind::Uncached { revisions } => {
                for (n, step) in revisions.iter().enumerate() {
                    writeln!(self.f, "REVISION {n}: {:?}", step.result)?;
                    self.nested(|this| this.format_evaluation_step(step))?;
                }
                writeln!(self.f, "RESULT: {:?}", goal.result)
            }
        }?;

        if goal.returned_goals.len() > 0 {
            writeln!(self.f, "NESTED GOALS ADDED TO CALLER: [")?;
            self.nested(|this| {
                for goal in goal.returned_goals.iter() {
                    writeln!(this.f, "ADDED GOAL: {:?},", goal)?;
                }
                Ok(())
            })?;

            writeln!(self.f, "]")?;
        }

        Ok(())
    }

    pub(super) fn format_evaluation_step(
        &mut self,
        evaluation_step: &GoalEvaluationStep<'_>,
    ) -> std::fmt::Result {
        writeln!(self.f, "INSTANTIATED: {:?}", evaluation_step.instantiated_goal)?;

        for candidate in &evaluation_step.candidates {
            self.nested(|this| this.format_candidate(candidate))?;
        }
        for nested in &evaluation_step.nested_goal_evaluations {
            self.nested(|this| this.format_nested_goal_evaluation(nested))?;
        }

        Ok(())
    }

    pub(super) fn format_candidate(&mut self, candidate: &GoalCandidate<'_>) -> std::fmt::Result {
        match &candidate.kind {
            CandidateKind::NormalizedSelfTyAssembly => {
                writeln!(self.f, "NORMALIZING SELF TY FOR ASSEMBLY:")
            }
            CandidateKind::DynUpcastingAssembly => {
                writeln!(self.f, "ASSEMBLING CANDIDATES FOR DYN UPCASTING:")
            }
            CandidateKind::Candidate { name, result } => {
                writeln!(self.f, "CANDIDATE {}: {:?}", name, result)
            }
        }?;

        self.nested(|this| {
            for candidate in &candidate.candidates {
                this.format_candidate(candidate)?;
            }
            for nested in &candidate.nested_goal_evaluations {
                this.format_nested_goal_evaluation(nested)?;
            }
            Ok(())
        })
    }

    pub(super) fn format_nested_goal_evaluation(
        &mut self,
        nested_goal_evaluation: &AddedGoalsEvaluation<'_>,
    ) -> std::fmt::Result {
        writeln!(self.f, "TRY_EVALUATE_ADDED_GOALS: {:?}", nested_goal_evaluation.result)?;

        for (n, revision) in nested_goal_evaluation.evaluations.iter().enumerate() {
            writeln!(self.f, "REVISION {n}")?;
            self.nested(|this| {
                for goal_evaluation in revision {
                    this.format_goal_evaluation(goal_evaluation)?;
                }
                Ok(())
            })?;
        }

        Ok(())
    }
}
