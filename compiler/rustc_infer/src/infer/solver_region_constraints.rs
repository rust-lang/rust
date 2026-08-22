use rustc_middle::ty::TyCtxt;
use rustc_span::Span;
use rustc_type_ir::region_constraint::RegionConstraint;
use tracing::instrument;

use super::InferCtxt;

pub type SolverRegionConstraint<'tcx> = RegionConstraint<TyCtxt<'tcx>, Span>;

#[derive(Clone, Debug)]
pub(crate) struct SolverRegionConstraintStorage<'tcx>(SolverRegionConstraint<'tcx>);

impl<'tcx> SolverRegionConstraintStorage<'tcx> {
    pub(crate) fn new() -> Self {
        Self(SolverRegionConstraint::new_true())
    }

    pub(crate) fn get_constraint(&self) -> SolverRegionConstraint<'tcx> {
        self.0.clone()
    }

    fn take(&mut self) -> SolverRegionConstraint<'tcx> {
        core::mem::replace(&mut self.0, SolverRegionConstraint::new_true())
    }

    #[instrument(level = "debug", skip(self))]
    pub(crate) fn overwrite(&mut self, constraint: SolverRegionConstraint<'tcx>) {
        self.0 = constraint;
    }
}

impl<'tcx> InferCtxt<'tcx> {
    pub(crate) fn clone_solver_region_constraints(&self) -> RegionConstraint<TyCtxt<'tcx>> {
        self.get_solver_region_constraint().without_spans()
    }

    /// Runs `op` with an empty solver-region-constraint store, restores the
    /// caller's constraints, and returns the constraints produced by `op`.
    pub fn with_fresh_solver_region_constraints<R>(
        &self,
        op: impl FnOnce() -> R,
    ) -> (R, RegionConstraint<TyCtxt<'tcx>>) {
        assert!(!self.in_snapshot(), "cannot isolate solver region constraints in a snapshot");

        let previous = self.inner.borrow_mut().solver_region_constraint_storage.take();
        let result = op();
        let current = self.inner.borrow_mut().solver_region_constraint_storage.take();
        self.inner.borrow_mut().solver_region_constraint_storage.overwrite(previous);

        (result, current.without_spans())
    }
}

#[cfg(test)]
mod tests;
