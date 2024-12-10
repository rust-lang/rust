use rustc_data_structures::captures::Captures;
use rustc_middle::mir::coverage::{
    CovTerm, CoverageIdsInfo, Expression, FunctionCoverageInfo, Mapping, MappingKind, Op,
    SourceRegion,
};

use crate::coverageinfo::ffi::{Counter, CounterExpression, ExprKind};

pub(crate) struct FunctionCoverage<'tcx> {
    pub(crate) function_coverage_info: &'tcx FunctionCoverageInfo,
    /// If `None`, the corresponding function is unused.
    ids_info: Option<&'tcx CoverageIdsInfo>,
}

impl<'tcx> FunctionCoverage<'tcx> {
    pub(crate) fn new_used(
        function_coverage_info: &'tcx FunctionCoverageInfo,
        ids_info: &'tcx CoverageIdsInfo,
    ) -> Self {
        Self { function_coverage_info, ids_info: Some(ids_info) }
    }

    pub(crate) fn new_unused(function_coverage_info: &'tcx FunctionCoverageInfo) -> Self {
        Self { function_coverage_info, ids_info: None }
    }

    /// Returns true for a used (called) function, and false for an unused function.
    pub(crate) fn is_used(&self) -> bool {
        self.ids_info.is_some()
    }

    /// Return the source hash, generated from the HIR node structure, and used to indicate whether
    /// or not the source code structure changed between different compilations.
    pub(crate) fn source_hash(&self) -> u64 {
        if self.is_used() { self.function_coverage_info.function_source_hash } else { 0 }
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
        match self.ids_info {
            Some(ids_info) => ids_info.is_zero_term(term),
            // This function is unused, so all coverage counters/expressions are zero.
            None => true,
        }
    }
}
