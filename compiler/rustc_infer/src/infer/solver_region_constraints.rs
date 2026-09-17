use rustc_middle::ty::TyCtxt;
use rustc_span::Span;
use tracing::instrument;

pub type SolverRegionConstraint<'tcx> =
    rustc_type_ir::region_constraint::RegionConstraint<TyCtxt<'tcx>, Span>;

#[derive(Clone, Debug)]
// Combining independent goals during accumulation forms a Cartesian product
// of their alternatives. Region checking can consume these batches directly.
pub(crate) struct SolverRegionConstraintStorage<'tcx>(Vec<SolverRegionConstraint<'tcx>>);

impl<'tcx> SolverRegionConstraintStorage<'tcx> {
    pub(crate) fn new() -> Self {
        Self(Vec::new())
    }

    pub(crate) fn get_constraint(&self) -> SolverRegionConstraint<'tcx> {
        match self.0.as_slice() {
            [] => SolverRegionConstraint::new_true(),
            [constraint] => constraint.clone(),
            constraints => {
                constraints.iter().cloned().reduce(SolverRegionConstraint::build_and).unwrap()
            }
        }
    }

    pub(crate) fn constraints(&self) -> &[SolverRegionConstraint<'tcx>] {
        &self.0
    }

    pub(crate) fn push(&mut self, constraint: SolverRegionConstraint<'tcx>) {
        self.0.push(constraint);
    }

    pub(crate) fn pop(&mut self) -> Option<SolverRegionConstraint<'tcx>> {
        self.0.pop()
    }

    pub(crate) fn take(&mut self) -> Vec<SolverRegionConstraint<'tcx>> {
        std::mem::take(&mut self.0)
    }

    #[instrument(level = "debug", skip(self))]
    pub(crate) fn overwrite(&mut self, constraints: Vec<SolverRegionConstraint<'tcx>>) {
        self.0 = constraints;
    }
}

#[cfg(test)]
mod tests;
