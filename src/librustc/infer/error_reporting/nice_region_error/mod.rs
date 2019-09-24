use crate::infer::InferCtxt;
use crate::infer::lexical_region_resolve::RegionResolutionError;
use crate::infer::lexical_region_resolve::RegionResolutionError::*;
use crate::ty::{self, TyCtxt};
use crate::util::common::ErrorReported;
use errors::DiagnosticBuilder;
use syntax::source_map::Span;

mod different_lifetimes;
mod find_anon_type;
mod named_anon_conflict;
mod placeholder_error;
mod outlives_closure;
mod static_impl_trait;
mod util;

impl<'cx, 'tcx> InferCtxt<'cx, 'tcx> {
    pub fn try_report_nice_region_error(&self, error: &RegionResolutionError<'tcx>) -> bool {
        match *error {
            ConcreteFailure(..) | SubSupConflict(..) => {}
            _ => return false,  // inapplicable
        }

        if let Some(tables) = self.in_progress_tables {
            let tables = tables.borrow();
            NiceRegionError::new(self, error.clone(), Some(&tables)).try_report().is_some()
        } else {
            NiceRegionError::new(self, error.clone(), None).try_report().is_some()
        }
    }
}

pub struct NiceRegionError<'cx, 'tcx> {
    infcx: &'cx InferCtxt<'cx, 'tcx>,
    error: Option<RegionResolutionError<'tcx>>,
    regions: Option<(Span, ty::Region<'tcx>, ty::Region<'tcx>)>,
    tables: Option<&'cx ty::TypeckTables<'tcx>>,
}

impl<'cx, 'tcx> NiceRegionError<'cx, 'tcx> {
    pub fn new(
        infcx: &'cx InferCtxt<'cx, 'tcx>,
        error: RegionResolutionError<'tcx>,
        tables: Option<&'cx ty::TypeckTables<'tcx>>,
    ) -> Self {
        Self { infcx, error: Some(error), regions: None, tables }
    }

    pub fn new_from_span(
        infcx: &'cx InferCtxt<'cx, 'tcx>,
        span: Span,
        sub: ty::Region<'tcx>,
        sup: ty::Region<'tcx>,
        tables: Option<&'cx ty::TypeckTables<'tcx>>,
    ) -> Self {
        Self { infcx, error: None, regions: Some((span, sub, sup)), tables }
    }

    fn tcx(&self) -> TyCtxt<'tcx> {
        self.infcx.tcx
    }

    pub fn try_report_from_nll(&self) -> Option<DiagnosticBuilder<'cx>> {
        // Due to the improved diagnostics returned by the MIR borrow checker, only a subset of
        // the nice region errors are required when running under the MIR borrow checker.
        self.try_report_named_anon_conflict()
            .or_else(|| self.try_report_placeholder_conflict())
    }

    pub fn try_report(&self) -> Option<ErrorReported> {
        self.try_report_from_nll()
            .map(|mut diag| { diag.emit(); ErrorReported })
            .or_else(|| self.try_report_anon_anon_conflict())
            .or_else(|| self.try_report_outlives_closure())
            .or_else(|| self.try_report_static_impl_trait())
    }

    pub fn get_regions(&self) -> (Span, ty::Region<'tcx>, ty::Region<'tcx>) {
        match (&self.error, self.regions) {
            (Some(ConcreteFailure(origin, sub, sup)), None) => (origin.span(), sub, sup),
            (Some(SubSupConflict(_, _, origin, sub, _, sup)), None) => (origin.span(), sub, sup),
            (None, Some((span, sub, sup))) => (span, sub, sup),
            (Some(_), Some(_)) => panic!("incorrectly built NiceRegionError"),
            _ => panic!("trying to report on an incorrect lifetime failure"),
        }
    }
}
