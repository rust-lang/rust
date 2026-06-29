use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_span::Span;
use rustc_type_ir::region_constraint::RegionConstraint as UnspannedRegionConstraint;
use tracing::instrument;

/// A solver region constraint together with the span that caused each atomic constraint.
///
/// Solver query responses use [`UnspannedRegionConstraint`] so that source locations do not
/// participate in candidate equality or caching. Spans are attached when those responses are
/// applied to an inference context.
#[derive(Clone, Debug)]
pub(crate) enum SolverRegionConstraint<'tcx> {
    Ambiguity(Span),
    RegionOutlives(ty::Region<'tcx>, ty::Region<'tcx>, Span),
    AliasTyOutlivesViaEnv(ty::Binder<'tcx, (ty::AliasTy<'tcx>, ty::Region<'tcx>)>, Span),
    PlaceholderTyOutlives(Ty<'tcx>, ty::Region<'tcx>, Span),
    And(Box<[SolverRegionConstraint<'tcx>]>),
    Or(Box<[SolverRegionConstraint<'tcx>]>),
}

impl<'tcx> SolverRegionConstraint<'tcx> {
    fn with_span(
        constraint: UnspannedRegionConstraint<TyCtxt<'tcx>>,
        span: Span,
    ) -> SolverRegionConstraint<'tcx> {
        use rustc_type_ir::region_constraint::RegionConstraint::*;

        match constraint {
            Ambiguity => Self::Ambiguity(span),
            RegionOutlives(a, b) => Self::RegionOutlives(a, b, span),
            AliasTyOutlivesViaEnv(outlives) => Self::AliasTyOutlivesViaEnv(outlives, span),
            PlaceholderTyOutlives(ty, region) => Self::PlaceholderTyOutlives(ty, region, span),
            And(constraints) => {
                Self::And(constraints.into_iter().map(|c| Self::with_span(c, span)).collect())
            }
            Or(constraints) => {
                Self::Or(constraints.into_iter().map(|c| Self::with_span(c, span)).collect())
            }
        }
    }

    fn without_spans(&self) -> UnspannedRegionConstraint<TyCtxt<'tcx>> {
        use rustc_type_ir::region_constraint::RegionConstraint::*;

        match self {
            Self::Ambiguity(_) => Ambiguity,
            Self::RegionOutlives(a, b, _) => RegionOutlives(*a, *b),
            Self::AliasTyOutlivesViaEnv(outlives, _) => AliasTyOutlivesViaEnv(*outlives),
            Self::PlaceholderTyOutlives(ty, region, _) => PlaceholderTyOutlives(*ty, *region),
            Self::And(constraints) => And(constraints.iter().map(Self::without_spans).collect()),
            Self::Or(constraints) => Or(constraints.iter().map(Self::without_spans).collect()),
        }
    }

    pub(crate) fn map_atomic_constraints(
        self,
        f: &mut impl FnMut(
            UnspannedRegionConstraint<TyCtxt<'tcx>>,
        ) -> UnspannedRegionConstraint<TyCtxt<'tcx>>,
    ) -> SolverRegionConstraint<'tcx> {
        let (constraint, span) = match self {
            Self::Ambiguity(span) => (UnspannedRegionConstraint::Ambiguity, span),
            Self::RegionOutlives(a, b, span) => {
                (UnspannedRegionConstraint::RegionOutlives(a, b), span)
            }
            Self::AliasTyOutlivesViaEnv(outlives, span) => {
                (UnspannedRegionConstraint::AliasTyOutlivesViaEnv(outlives), span)
            }
            Self::PlaceholderTyOutlives(ty, region, span) => {
                (UnspannedRegionConstraint::PlaceholderTyOutlives(ty, region), span)
            }
            Self::And(constraints) => {
                return Self::And(
                    constraints.into_iter().map(|c| c.map_atomic_constraints(f)).collect(),
                );
            }
            Self::Or(constraints) => {
                return Self::Or(
                    constraints.into_iter().map(|c| c.map_atomic_constraints(f)).collect(),
                );
            }
        };

        Self::with_span(f(constraint), span)
    }

    pub(crate) fn evaluate(self) -> SolverRegionConstraint<'tcx> {
        match self {
            Self::And(constraints) => {
                let mut evaluated = Vec::new();
                let mut ambiguity = None;
                for constraint in constraints {
                    let constraint = constraint.evaluate();
                    if constraint.is_false() {
                        return Self::Or(Box::new([]));
                    } else if let Self::Ambiguity(span) = constraint {
                        ambiguity.get_or_insert(span);
                    } else if !constraint.is_true() {
                        evaluated.push(constraint);
                    }
                }

                ambiguity.map_or_else(|| Self::And(evaluated.into_boxed_slice()), Self::Ambiguity)
            }
            Self::Or(constraints) => {
                let mut evaluated = Vec::new();
                let mut ambiguity = None;
                for constraint in constraints {
                    let constraint = constraint.evaluate();
                    if constraint.is_true() {
                        return Self::And(Box::new([]));
                    } else if let Self::Ambiguity(span) = constraint {
                        ambiguity.get_or_insert(span);
                    } else if !constraint.is_false() {
                        evaluated.push(constraint);
                    }
                }

                ambiguity.map_or_else(|| Self::Or(evaluated.into_boxed_slice()), Self::Ambiguity)
            }
            constraint => constraint,
        }
    }

    fn is_true(&self) -> bool {
        matches!(self, Self::And(constraints) if constraints.is_empty())
    }

    fn is_false(&self) -> bool {
        matches!(self, Self::Or(constraints) if constraints.is_empty())
    }
}

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
        self.0.without_spans()
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
        let constraint = SolverRegionConstraint::with_span(constraint, span);
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
        self.overwrite_spanned(SolverRegionConstraint::with_span(constraint, span));
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
