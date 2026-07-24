use rustc_middle::ty::TyCtxt;
use rustc_span::Span;
use tracing::instrument;

pub type SolverRegionConstraint<'tcx> =
    rustc_type_ir::region_constraint::CanonicalFormRegionConstraint<TyCtxt<'tcx>, Span>;

#[derive(Clone, Debug)]
pub(crate) struct SolverRegionConstraintStorage<'tcx>(SolverRegionConstraint<'tcx>);

impl<'tcx> SolverRegionConstraintStorage<'tcx> {
    pub(crate) fn new() -> Self {
        Self(SolverRegionConstraint::new_true())
    }

    pub(crate) fn get_constraint(&self) -> SolverRegionConstraint<'tcx> {
        self.0.clone()
    }

    #[instrument(level = "debug", skip(self))]
    pub(crate) fn overwrite(&mut self, constraint: SolverRegionConstraint<'tcx>) {
        self.0 = constraint;
    }
}

#[cfg(test)]
mod tests;
