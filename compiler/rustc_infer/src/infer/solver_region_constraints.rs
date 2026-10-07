use rustc_middle::ty::TyCtxt;
use rustc_span::Span;
use tracing::instrument;

use super::InferCtxt;

pub type SolverRegionConstraint<'tcx> =
    rustc_type_ir::region_constraint::RegionConstraint<TyCtxt<'tcx>, Span>;

#[derive(Clone, Debug)]
pub(crate) struct SolverRegionConstraintStorage<'tcx>(Option<SolverRegionConstraint<'tcx>>);

impl<'tcx> SolverRegionConstraintStorage<'tcx> {
    pub(crate) fn new() -> Self {
        Self(None)
    }

    pub(crate) fn get_constraint(&self) -> SolverRegionConstraint<'tcx> {
        match &self.0 {
            Some(v) => v.clone(),
            None => SolverRegionConstraint::new_true(),
        }
    }

    pub(crate) fn take(&mut self) -> Option<SolverRegionConstraint<'tcx>> {
        self.0.take()
    }

    #[instrument(level = "debug", skip(self))]
    pub(crate) fn overwrite(&mut self, constraint: SolverRegionConstraint<'tcx>) {
        self.0 = Some(constraint);
    }
}

impl<'tcx> InferCtxt<'tcx> {
    /// Drain completed query constraints without allocating an empty constraint.
    pub fn take_solver_region_constraints(
        &self,
    ) -> Option<Box<rustc_type_ir::region_constraint::RegionConstraint<TyCtxt<'tcx>>>> {
        assert!(!self.in_snapshot(), "cannot take solver region constraints in a snapshot");
        if !self.tcx.assumptions_on_binders() {
            return None;
        }
        self.inner
            .borrow_mut()
            .solver_region_constraint_storage
            .take()
            .map(|constraint| Box::new(constraint.without_spans()))
    }
}

#[cfg(test)]
mod tests;
