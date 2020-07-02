use crate::builder::Builder;
use crate::common::CodegenCx;
use log::debug;
use rustc_codegen_ssa::coverageinfo::map::*;
use rustc_codegen_ssa::traits::{CoverageInfoBuilderMethods, CoverageInfoMethods};
use rustc_data_structures::fx::FxHashMap;
use rustc_middle::ty::Instance;

use std::cell::RefCell;

/// A context object for maintaining all state needed by the coverageinfo module.
pub struct CrateCoverageContext<'tcx> {
    // Coverage region data for each instrumented function identified by DefId.
    pub(crate) coverage_regions: RefCell<FxHashMap<Instance<'tcx>, FunctionCoverageRegions>>,
}

impl<'tcx> CrateCoverageContext<'tcx> {
    pub fn new() -> Self {
        Self { coverage_regions: Default::default() }
    }
}

/// Generates and exports the Coverage Map.
// FIXME(richkadel): Actually generate and export the coverage map to LLVM.
// The current implementation is actually just debug messages to show the data is available.
pub fn finalize(cx: &CodegenCx<'_, '_>) {
    let coverage_regions = &*cx.coverage_context().coverage_regions.borrow();
    for instance in coverage_regions.keys() {
        let coverageinfo = cx.tcx.coverageinfo(instance.def_id());
        debug_assert!(coverageinfo.num_counters > 0);
        debug!(
            "Generate coverage map for: {:?}, hash: {}, num_counters: {}",
            instance, coverageinfo.hash, coverageinfo.num_counters
        );
        let function_coverage_regions = &coverage_regions[instance];
        for (index, region) in function_coverage_regions.indexed_regions() {
            match region.kind {
                CoverageKind::Counter => debug!(
                    "  Counter {}, for {}..{}",
                    index, region.coverage_span.start_byte_pos, region.coverage_span.end_byte_pos
                ),
                CoverageKind::CounterExpression(lhs, op, rhs) => debug!(
                    "  CounterExpression {} = {} {:?} {}, for {}..{}",
                    index,
                    lhs,
                    op,
                    rhs,
                    region.coverage_span.start_byte_pos,
                    region.coverage_span.end_byte_pos
                ),
            }
        }
        for unreachable in function_coverage_regions.unreachable_regions() {
            debug!(
                "  Unreachable code region: {}..{}",
                unreachable.start_byte_pos, unreachable.end_byte_pos
            );
        }
    }
}

impl CoverageInfoMethods for CodegenCx<'ll, 'tcx> {
    fn coverageinfo_finalize(&self) {
        finalize(self)
    }
}

impl CoverageInfoBuilderMethods<'tcx> for Builder<'a, 'll, 'tcx> {
    fn add_counter_region(
        &mut self,
        instance: Instance<'tcx>,
        index: u32,
        start_byte_pos: u32,
        end_byte_pos: u32,
    ) {
        debug!(
            "adding counter to coverage map: instance={:?}, index={}, byte range {}..{}",
            instance, index, start_byte_pos, end_byte_pos,
        );
        let mut coverage_regions = self.coverage_context().coverage_regions.borrow_mut();
        coverage_regions.entry(instance).or_default().add_counter(
            index,
            start_byte_pos,
            end_byte_pos,
        );
    }

    fn add_counter_expression_region(
        &mut self,
        instance: Instance<'tcx>,
        index: u32,
        lhs: u32,
        op: CounterOp,
        rhs: u32,
        start_byte_pos: u32,
        end_byte_pos: u32,
    ) {
        debug!(
            "adding counter expression to coverage map: instance={:?}, index={}, {} {:?} {}, byte range {}..{}",
            instance, index, lhs, op, rhs, start_byte_pos, end_byte_pos,
        );
        let mut coverage_regions = self.coverage_context().coverage_regions.borrow_mut();
        coverage_regions.entry(instance).or_default().add_counter_expression(
            index,
            lhs,
            op,
            rhs,
            start_byte_pos,
            end_byte_pos,
        );
    }

    fn add_unreachable_region(
        &mut self,
        instance: Instance<'tcx>,
        start_byte_pos: u32,
        end_byte_pos: u32,
    ) {
        debug!(
            "adding unreachable code to coverage map: instance={:?}, byte range {}..{}",
            instance, start_byte_pos, end_byte_pos,
        );
        let mut coverage_regions = self.coverage_context().coverage_regions.borrow_mut();
        coverage_regions.entry(instance).or_default().add_unreachable(start_byte_pos, end_byte_pos);
    }
}
