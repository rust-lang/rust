use rustc_middle::ty::TyCtxt;
use rustc_span::Span;
use tracing::instrument;

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

    #[instrument(level = "debug", skip(self))]
    pub(crate) fn overwrite(&mut self, constraint: SolverRegionConstraint<'tcx>) {
        self.0 = Some(constraint);
    }
}

#[cfg(test)]
mod tests;
