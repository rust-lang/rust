use rustc_data_structures::captures::Captures;
use rustc_middle::mir::coverage::{
    CovTerm, CoverageIdsInfo, Expression, FunctionCoverageInfo, Mapping, MappingKind, Op,
    SourceRegion,
};
use rustc_middle::ty::Instance;
use tracing::debug;

use crate::coverageinfo::ffi::{Counter, CounterExpression, ExprKind};

/// Holds all of the coverage mapping data associated with a function instance,
/// collected during traversal of `Coverage` statements in the function's MIR.
#[derive(Debug)]
pub(crate) struct FunctionCoverageCollector<'tcx> {
    /// Coverage info that was attached to this function by the instrumentor.
    function_coverage_info: &'tcx FunctionCoverageInfo,
    ids_info: &'tcx CoverageIdsInfo,
    is_used: bool,
}

impl<'tcx> FunctionCoverageCollector<'tcx> {
    /// Creates a new set of coverage data for a used (called) function.
    pub(crate) fn new(
        instance: Instance<'tcx>,
        function_coverage_info: &'tcx FunctionCoverageInfo,
        ids_info: &'tcx CoverageIdsInfo,
    ) -> Self {
        Self::create(instance, function_coverage_info, ids_info, true)
    }

    /// Creates a new set of coverage data for an unused (never called) function.
    pub(crate) fn unused(
        instance: Instance<'tcx>,
        function_coverage_info: &'tcx FunctionCoverageInfo,
        ids_info: &'tcx CoverageIdsInfo,
    ) -> Self {
        Self::create(instance, function_coverage_info, ids_info, false)
    }

    fn create(
        instance: Instance<'tcx>,
        function_coverage_info: &'tcx FunctionCoverageInfo,
        ids_info: &'tcx CoverageIdsInfo,
        is_used: bool,
    ) -> Self {
        let num_counters = function_coverage_info.num_counters;
        let num_expressions = function_coverage_info.expressions.len();
        debug!(
            "FunctionCoverage::create(instance={instance:?}) has \
            num_counters={num_counters}, num_expressions={num_expressions}, is_used={is_used}"
        );

        Self { function_coverage_info, ids_info, is_used }
    }

    pub(crate) fn into_finished(self) -> FunctionCoverage<'tcx> {
        let FunctionCoverageCollector { function_coverage_info, ids_info, is_used, .. } = self;

        FunctionCoverage { function_coverage_info, ids_info, is_used }
    }
}

pub(crate) struct FunctionCoverage<'tcx> {
    pub(crate) function_coverage_info: &'tcx FunctionCoverageInfo,
    ids_info: &'tcx CoverageIdsInfo,
    is_used: bool,
}

impl<'tcx> FunctionCoverage<'tcx> {
    /// Returns true for a used (called) function, and false for an unused function.
    pub(crate) fn is_used(&self) -> bool {
        self.is_used
    }

    /// Return the source hash, generated from the HIR node structure, and used to indicate whether
    /// or not the source code structure changed between different compilations.
    pub(crate) fn source_hash(&self) -> u64 {
        if self.is_used { self.function_coverage_info.function_source_hash } else { 0 }
    }

    /// Convert this function's coverage expression data into a form that can be
    /// passed through FFI to LLVM.
    pub(crate) fn counter_expressions(
        &self,
    ) -> impl Iterator<Item = CounterExpression> + ExactSizeIterator + Captures<'_> {
        // We know that LLVM will optimize out any unused expressions before
        // producing the final coverage map, so there's no need to do the same
        // thing on the Rust side unless we're confident we can do much better.
        // (See `CounterExpressionsMinimizer` in `CoverageMappingWriter.cpp`.)

        self.function_coverage_info.expressions.iter().map(move |&Expression { lhs, op, rhs }| {
            CounterExpression {
                lhs: self.counter_for_term(lhs),
                kind: match op {
                    Op::Add => ExprKind::Add,
                    Op::Subtract => ExprKind::Subtract,
                },
                rhs: self.counter_for_term(rhs),
            }
        })
    }

    /// Converts this function's coverage mappings into an intermediate form
    /// that will be used by `mapgen` when preparing for FFI.
    pub(crate) fn counter_regions(
        &self,
    ) -> impl Iterator<Item = (MappingKind, &SourceRegion)> + ExactSizeIterator {
        self.function_coverage_info.mappings.iter().map(move |mapping| {
            let Mapping { kind, source_region } = mapping;
            let kind =
                kind.map_terms(|term| if self.is_zero_term(term) { CovTerm::Zero } else { term });
            (kind, source_region)
        })
    }

    fn counter_for_term(&self, term: CovTerm) -> Counter {
        if self.is_zero_term(term) { Counter::ZERO } else { Counter::from_term(term) }
    }

    fn is_zero_term(&self, term: CovTerm) -> bool {
        !self.is_used || self.ids_info.is_zero_term(term)
    }
}
