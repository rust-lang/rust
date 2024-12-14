use rustc_middle::mir::coverage::{CoverageIdsInfo, FunctionCoverageInfo};

pub(crate) struct FunctionCoverage<'tcx> {
    #[expect(unused)] // This whole file gets deleted later in the same PR.
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
}
