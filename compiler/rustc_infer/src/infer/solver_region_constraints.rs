use rustc_middle::ty::TyCtxt;
use rustc_type_ir::region_constraint::SpannedRegionConstraint;
use tracing::instrument;

pub type SolverRegionConstraint<'tcx> = SpannedRegionConstraint<TyCtxt<'tcx>>;

#[derive(Clone, Debug)]
pub(crate) struct SolverRegionConstraintStorage<'tcx>(SolverRegionConstraint<'tcx>);

impl<'tcx> SolverRegionConstraintStorage<'tcx> {
    pub(crate) fn new() -> Self {
        Self(SolverRegionConstraint::And(Box::new([])))
    }

    pub(crate) fn get_constraint(&self) -> SolverRegionConstraint<'tcx> {
        self.0.clone()
    }

    pub(crate) fn is_and(&self) -> bool {
        self.0.is_and()
    }

    pub(crate) fn pop(&mut self, previous_was_and: bool) -> Option<SolverRegionConstraint<'tcx>> {
        match &mut self.0 {
            SolverRegionConstraint::And(and) => {
                let mut and = core::mem::take(and).into_vec();
                let popped = and.pop()?;
                if previous_was_and {
                    self.0 = SolverRegionConstraint::And(and.into_boxed_slice());
                } else {
                    assert_eq!(and.len(), 1);
                    self.0 = and.pop().unwrap();
                }
                Some(popped)
            }
            _ => unreachable!(),
        }
    }

    #[instrument(level = "debug")]
    pub(crate) fn push(&mut self, constraint: SolverRegionConstraint<'tcx>) {
        match core::mem::replace(&mut self.0, SolverRegionConstraint::new_true()) {
            SolverRegionConstraint::And(and) => {
                let and =
                    and.into_iter().chain([constraint]).collect::<Vec<_>>().into_boxed_slice();
                self.0 = SolverRegionConstraint::And(and);
            }
            previous => {
                self.0 = SolverRegionConstraint::And(Box::new([previous, constraint]));
            }
        }
    }

    #[instrument(level = "debug", skip(self))]
    pub(crate) fn overwrite(&mut self, constraint: SolverRegionConstraint<'tcx>) {
        self.0 = constraint;
    }
}

#[cfg(test)]
mod tests;
