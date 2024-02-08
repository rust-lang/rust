use rustc_middle::mir;
use rustc_middle::mir::coverage::BranchSpan;
use rustc_middle::ty::TyCtxt;
use rustc_span::def_id::LocalDefId;

pub(crate) struct HirBranchInfoBuilder {
    num_block_markers: usize,
    branch_spans: Vec<BranchSpan>,
}

impl HirBranchInfoBuilder {
    /// Creates a new branch info builder, but only if branch coverage instrumentation
    /// is enabled and `def_id` represents a function that is eligible for coverage.
    pub(crate) fn new_if_enabled(tcx: TyCtxt<'_>, def_id: LocalDefId) -> Option<Self> {
        if tcx.sess.instrument_coverage_branch() && tcx.is_eligible_for_coverage(def_id) {
            Some(Self {
                // (placeholder)
                num_block_markers: 0,
                branch_spans: vec![],
            })
        } else {
            None
        }
    }

    pub(crate) fn into_done(self) -> Option<Box<mir::coverage::HirBranchInfo>> {
        let Self { num_block_markers, branch_spans } = self;

        if num_block_markers == 0 {
            assert!(branch_spans.is_empty());
            return None;
        }

        Some(Box::new(mir::coverage::HirBranchInfo { num_block_markers, branch_spans }))
    }
}
