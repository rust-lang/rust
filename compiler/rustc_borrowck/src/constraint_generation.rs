#![deny(rustc::untranslatable_diagnostic)]
#![deny(rustc::diagnostic_outside_of_impl)]
use rustc_middle::mir::visit::{TyContext, Visitor};
use rustc_middle::mir::{Body, Location, SourceInfo};
use rustc_middle::ty::visit::TypeVisitable;
use rustc_middle::ty::{GenericArgsRef, Region, Ty, TyCtxt};

use crate::region_infer::values::LivenessValues;

pub(super) fn generate_constraints<'tcx>(
    tcx: TyCtxt<'tcx>,
    liveness_constraints: &mut LivenessValues,
    body: &Body<'tcx>,
) {
    let mut cg = ConstraintGeneration { tcx, liveness_constraints };
    for (bb, data) in body.basic_blocks.iter_enumerated() {
        cg.visit_basic_block_data(bb, data);
    }
}

/// 'cg = the duration of the constraint generation process itself.
struct ConstraintGeneration<'cg, 'tcx> {
    tcx: TyCtxt<'tcx>,
    liveness_constraints: &'cg mut LivenessValues,
}

impl<'cg, 'tcx> Visitor<'tcx> for ConstraintGeneration<'cg, 'tcx> {
    /// We sometimes have `args` within an rvalue, or within a
    /// call. Make them live at the location where they appear.
    fn visit_args(&mut self, args: &GenericArgsRef<'tcx>, location: Location) {
        self.record_regions_live_at(*args, location);
        self.super_args(args);
    }

    /// We sometimes have `region`s within an rvalue, or within a
    /// call. Make them live at the location where they appear.
    fn visit_region(&mut self, region: Region<'tcx>, location: Location) {
        self.record_regions_live_at(region, location);
        self.super_region(region);
    }

    /// We sometimes have `ty`s within an rvalue, or within a
    /// call. Make them live at the location where they appear.
    fn visit_ty(&mut self, ty: Ty<'tcx>, ty_context: TyContext) {
        match ty_context {
            TyContext::ReturnTy(SourceInfo { span, .. })
            | TyContext::YieldTy(SourceInfo { span, .. })
            | TyContext::UserTy(span)
            | TyContext::LocalDecl { source_info: SourceInfo { span, .. }, .. } => {
                span_bug!(span, "should not be visiting outside of the CFG: {:?}", ty_context);
            }
            TyContext::Location(location) => {
                self.record_regions_live_at(ty, location);
            }
        }

        self.super_ty(ty);
    }
}

impl<'cx, 'tcx> ConstraintGeneration<'cx, 'tcx> {
    /// Some variable is "regular live" at `location` -- i.e., it may be used later. This means that
    /// all regions appearing in the type of `value` must be live at `location`.
    fn record_regions_live_at<T>(&mut self, value: T, location: Location)
    where
        T: TypeVisitable<TyCtxt<'tcx>>,
    {
        debug!("add_regular_live_constraint(value={:?}, location={:?})", value, location);
        self.tcx.for_each_free_region(&value, |live_region| {
            let live_region_vid = live_region.as_var();
            self.liveness_constraints.add_location(live_region_vid, location);
        });
    }
}
