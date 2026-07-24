use rustc_middle::ty::TyCtxt;
use rustc_span::Span;
use rustc_type_ir::region_constraint::{
    RegionConstraint as UnspannedRegionConstraint, SpannedRegionConstraint,
};
use tracing::instrument;

pub(crate) type SolverRegionConstraint<'tcx> = SpannedRegionConstraint<TyCtxt<'tcx>>;

#[derive(Clone, Debug)]
pub(crate) struct SolverRegionConstraintStorage<'tcx>(SolverRegionConstraint<'tcx>);

impl<'tcx> SolverRegionConstraintStorage<'tcx> {
    pub(crate) fn new() -> Self {
        Self(SolverRegionConstraint::And(Box::new([])))
    }

    pub(crate) fn get_constraint(&self) -> SolverRegionConstraint<'tcx> {
        self.0.clone()
    }

    pub(crate) fn get_unspanned_constraint(&self) -> UnspannedRegionConstraint<TyCtxt<'tcx>> {
        self.0.clone().without_spans()
    }

    pub(crate) fn pop(&mut self) -> Option<SolverRegionConstraint<'tcx>> {
        match &mut self.0 {
            SolverRegionConstraint::And(and) => {
                let mut and = core::mem::take(and).into_vec();
                let popped = and.pop()?;
                self.0 = SolverRegionConstraint::And(and.into_boxed_slice());
                Some(popped)
            }
            _ => unreachable!(),
        }
    }

    #[instrument(level = "debug")]
    pub(crate) fn push(&mut self, constraint: UnspannedRegionConstraint<TyCtxt<'tcx>>, span: Span) {
        let constraint = constraint.with_span(span);
        match &mut self.0 {
            SolverRegionConstraint::And(and) => {
                let and = core::mem::take(and)
                    .into_iter()
                    .chain([constraint])
                    .collect::<Vec<_>>()
                    .into_boxed_slice();
                self.0 = SolverRegionConstraint::And(and);
            }
            _ => unreachable!(),
        }
    }

    #[instrument(level = "debug", skip(self))]
    pub(crate) fn overwrite(
        &mut self,
        constraint: UnspannedRegionConstraint<TyCtxt<'tcx>>,
        span: Span,
    ) {
        self.overwrite_spanned(constraint.with_span(span));
    }

    pub(crate) fn overwrite_spanned(&mut self, constraint: SolverRegionConstraint<'tcx>) {
        if matches!(constraint, SolverRegionConstraint::And(_)) {
            self.0 = constraint;
        } else {
            self.0 = SolverRegionConstraint::And(vec![constraint].into_boxed_slice());
        }
    }
}

#[cfg(test)]
mod tests;
