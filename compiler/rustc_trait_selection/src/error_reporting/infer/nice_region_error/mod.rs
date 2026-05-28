use rustc_errors::{Diag, ErrorGuaranteed};
use rustc_hir::def_id::LocalDefId;
use rustc_middle::ty::{self, TyCtxt};
use rustc_span::Span;

use crate::error_reporting::TypeErrCtxt;
use crate::infer::RegionResolutionError;

mod different_lifetimes;
pub mod find_anon_type;
mod mismatched_static_lifetime;
mod named_anon_conflict;
pub(crate) mod placeholder_error;
mod placeholder_relation;
mod static_impl_trait;
mod trait_impl_difference;
mod util;

pub use different_lifetimes::suggest_adding_lifetime_params;
pub use find_anon_type::find_anon_type;
pub use static_impl_trait::{HirTraitObjectVisitor, TraitObjectVisitor, suggest_new_region_bound};
pub use util::find_param_with_region;

impl<'cx, 'tcx> TypeErrCtxt<'cx, 'tcx> {
    pub fn try_report_nice_region_error(
        &'cx self,
        generic_param_scope: LocalDefId,
        error: &RegionResolutionError<'tcx>,
    ) -> Option<ErrorGuaranteed> {
        NiceRegionError::new(self, generic_param_scope, error.clone()).try_report()
    }
}

pub struct NiceRegionError<'cx, 'tcx> {
    cx: &'cx TypeErrCtxt<'cx, 'tcx>,
    /// The innermost definition that introduces generic parameters that may be involved in
    /// the region errors we are dealing with.
    generic_param_scope: LocalDefId,
    error: Option<RegionResolutionError<'tcx>>,
    regions: Option<(Span, ty::Region<'tcx>, ty::Region<'tcx>)>,
}

impl<'cx, 'tcx> NiceRegionError<'cx, 'tcx> {
    pub fn new(
        cx: &'cx TypeErrCtxt<'cx, 'tcx>,
        generic_param_scope: LocalDefId,
        error: RegionResolutionError<'tcx>,
    ) -> Self {
        Self { cx, error: Some(error), regions: None, generic_param_scope }
    }

    pub fn new_from_span(
        cx: &'cx TypeErrCtxt<'cx, 'tcx>,
        generic_param_scope: LocalDefId,
        span: Span,
        sub: ty::Region<'tcx>,
        sup: ty::Region<'tcx>,
    ) -> Self {
        Self { cx, error: None, regions: Some((span, sub, sup)), generic_param_scope }
    }

    fn tcx(&self) -> TyCtxt<'tcx> {
        self.cx.tcx
    }

    pub fn try_report_from_nll(&self) -> Option<Diag<'tcx>> {
        // Due to the improved diagnostics returned by the MIR borrow checker, only a subset of
        // the nice region errors are required when running under the MIR borrow checker.
        self.try_report_named_anon_conflict()
            .or_else(|| self.try_report_placeholder_conflict())
            .or_else(|| self.try_report_placeholder_relation())
    }

    pub fn try_report(&self) -> Option<ErrorGuaranteed> {
        self.try_report_from_nll()
            .map(|diag| diag.emit())
            .or_else(|| self.try_report_impl_not_conforming_to_trait())
            .or_else(|| self.try_report_anon_anon_conflict())
            .or_else(|| self.try_report_static_impl_trait())
            .or_else(|| self.try_report_mismatched_static_lifetime())
    }

    pub(super) fn regions(&self) -> Option<(Span, ty::Region<'tcx>, ty::Region<'tcx>)> {
        match (&self.error, self.regions) {
            (Some(RegionResolutionError::ConcreteFailure(origin, sub, sup)), None) => {
                Some((origin.span(), *sub, *sup))
            }
            (Some(RegionResolutionError::SubSupConflict(_, _, origin, sub, _, sup, _)), None) => {
                Some((origin.span(), *sub, *sup))
            }
            (None, Some((span, sub, sup))) => Some((span, sub, sup)),
            _ => None,
        }
    }
}
