use std::collections::BTreeSet;

use rustc_data_structures::graph::DirectedGraph;
use rustc_index::IndexVec;
use rustc_index::bit_set::BitSet;
use rustc_middle::mir::coverage::{
    BlockMarkerId, BranchSpan, ConditionInfo, CoverageInfoHi, CoverageKind,
};
use rustc_middle::mir::{self, BasicBlock, StatementKind};
use rustc_middle::ty::TyCtxt;
use rustc_span::Span;

use crate::coverage::ExtractedHirInfo;
use crate::coverage::graph::{BasicCoverageBlock, CoverageGraph, START_BCB};
use crate::coverage::spans::extract_refined_covspans;
use crate::coverage::unexpand::unexpand_into_body_span;

/// Associates an ordinary executable code span with its corresponding BCB.
#[derive(Debug)]
pub(super) struct CodeMapping {
    pub(super) span: Span,
    pub(super) bcb: BasicCoverageBlock,
}

/// This is separate from [`MCDCBranch`] to help prepare for larger changes
/// that will be needed for improved branch coverage in the future.
/// (See <https://github.com/rust-lang/rust/pull/124217>.)
#[derive(Debug)]
pub(super) struct BranchPair {
    pub(super) span: Span,
    pub(super) true_bcb: BasicCoverageBlock,
    pub(super) false_bcb: BasicCoverageBlock,
}

/// Associates an MC/DC branch span with condition info besides fields for normal branch.
#[derive(Debug)]
pub(super) struct MCDCBranch {
    pub(super) span: Span,
    pub(super) true_bcb: BasicCoverageBlock,
    pub(super) false_bcb: BasicCoverageBlock,
    /// If `None`, this actually represents a normal branch mapping inserted
    /// for code that was too complex for MC/DC.
    pub(super) condition_info: Option<ConditionInfo>,
    pub(super) decision_depth: u16,
}

/// Associates an MC/DC decision with its join BCBs.
#[derive(Debug)]
pub(super) struct MCDCDecision {
    pub(super) span: Span,
    pub(super) end_bcbs: BTreeSet<BasicCoverageBlock>,
    pub(super) bitmap_idx: u32,
    pub(super) num_conditions: u16,
    pub(super) decision_depth: u16,
}

#[derive(Default)]
pub(super) struct ExtractedMappings {
    /// Store our own copy of [`CoverageGraph::num_nodes`], so that we don't
    /// need access to the whole graph when allocating per-BCB data. This is
    /// only public so that other code can still use exhaustive destructuring.
    pub(super) num_bcbs: usize,
    pub(super) code_mappings: Vec<CodeMapping>,
    pub(super) branch_pairs: Vec<BranchPair>,
    pub(super) mcdc_bitmap_bytes: u32,
    pub(super) mcdc_branches: Vec<MCDCBranch>,
    pub(super) mcdc_decisions: Vec<MCDCDecision>,
}

/// Extracts coverage-relevant spans from MIR, and associates them with
/// their corresponding BCBs.
pub(super) fn extract_all_mapping_info_from_mir<'tcx>(
    tcx: TyCtxt<'tcx>,
    mir_body: &mir::Body<'tcx>,
    hir_info: &ExtractedHirInfo,
    basic_coverage_blocks: &CoverageGraph,
) -> ExtractedMappings {
    let mut code_mappings = vec![];
    let mut branch_pairs = vec![];
    let mut mcdc_bitmap_bytes = 0;
    let mut mcdc_branches = vec![];
    let mut mcdc_decisions = vec![];

    if hir_info.is_async_fn || tcx.sess.coverage_no_mir_spans() {
        // An async function desugars into a function that returns a future,
        // with the user code wrapped in a closure. Any spans in the desugared
        // outer function will be unhelpful, so just keep the signature span
        // and ignore all of the spans in the MIR body.
        //
        // When debugging flag `-Zcoverage-options=no-mir-spans` is set, we need
        // to give the same treatment to _all_ functions, because `llvm-cov`
        // seems to ignore functions that don't have any ordinary code spans.
        if let Some(span) = hir_info.fn_sig_span_extended {
            code_mappings.push(CodeMapping { span, bcb: START_BCB });
        }
    } else {
        // Extract coverage spans from MIR statements/terminators as normal.
        extract_refined_covspans(mir_body, hir_info, basic_coverage_blocks, &mut code_mappings);
    }

    branch_pairs.extend(extract_branch_pairs(mir_body, hir_info, basic_coverage_blocks));

    extract_mcdc_mappings(
        mir_body,
        hir_info.body_span,
        basic_coverage_blocks,
        &mut mcdc_bitmap_bytes,
        &mut mcdc_branches,
        &mut mcdc_decisions,
    );

    ExtractedMappings {
        num_bcbs: basic_coverage_blocks.num_nodes(),
        code_mappings,
        branch_pairs,
        mcdc_bitmap_bytes,
        mcdc_branches,
        mcdc_decisions,
    }
}

impl ExtractedMappings {
    pub(super) fn all_bcbs_with_counter_mappings(&self) -> BitSet<BasicCoverageBlock> {
        // Fully destructure self to make sure we don't miss any fields that have mappings.
        let Self {
            num_bcbs,
            code_mappings,
            branch_pairs,
            mcdc_bitmap_bytes: _,
            mcdc_branches,
            mcdc_decisions,
        } = self;

        // Identify which BCBs have one or more mappings.
        let mut bcbs_with_counter_mappings = BitSet::new_empty(*num_bcbs);
        let mut insert = |bcb| {
            bcbs_with_counter_mappings.insert(bcb);
        };

        for &CodeMapping { span: _, bcb } in code_mappings {
            insert(bcb);
        }
        for &BranchPair { true_bcb, false_bcb, .. } in branch_pairs {
            insert(true_bcb);
            insert(false_bcb);
        }
        for &MCDCBranch { true_bcb, false_bcb, .. } in mcdc_branches {
            insert(true_bcb);
            insert(false_bcb);
        }

        // MC/DC decisions refer to BCBs, but don't require those BCBs to have counters.
        if bcbs_with_counter_mappings.is_empty() {
            debug_assert!(
                mcdc_decisions.is_empty(),
                "A function with no counter mappings shouldn't have any decisions: {mcdc_decisions:?}",
            );
        }

        bcbs_with_counter_mappings
    }

    /// Returns the set of BCBs that have one or more `Code` mappings.
    pub(super) fn bcbs_with_ordinary_code_mappings(&self) -> BitSet<BasicCoverageBlock> {
        let mut bcbs = BitSet::new_empty(self.num_bcbs);
        for &CodeMapping { span: _, bcb } in &self.code_mappings {
            bcbs.insert(bcb);
        }
        bcbs
    }
}

fn resolve_block_markers(
    coverage_info_hi: &CoverageInfoHi,
    mir_body: &mir::Body<'_>,
) -> IndexVec<BlockMarkerId, Option<BasicBlock>> {
    let mut block_markers = IndexVec::<BlockMarkerId, Option<BasicBlock>>::from_elem_n(
        None,
        coverage_info_hi.num_block_markers,
    );

    // Fill out the mapping from block marker IDs to their enclosing blocks.
    for (bb, data) in mir_body.basic_blocks.iter_enumerated() {
        for statement in &data.statements {
            if let StatementKind::Coverage(CoverageKind::BlockMarker { id }) = statement.kind {
                block_markers[id] = Some(bb);
            }
        }
    }

    block_markers
}

// FIXME: There is currently a lot of redundancy between
// `extract_branch_pairs` and `extract_mcdc_mappings`. This is needed so
// that they can each be modified without interfering with the other, but in
// the long term we should try to bring them together again when branch coverage
// and MC/DC coverage support are more mature.

pub(super) fn extract_branch_pairs(
    mir_body: &mir::Body<'_>,
    hir_info: &ExtractedHirInfo,
    basic_coverage_blocks: &CoverageGraph,
) -> Vec<BranchPair> {
    let Some(coverage_info_hi) = mir_body.coverage_info_hi.as_deref() else { return vec![] };

    let block_markers = resolve_block_markers(coverage_info_hi, mir_body);

    coverage_info_hi
        .branch_spans
        .iter()
        .filter_map(|&BranchSpan { span: raw_span, true_marker, false_marker }| {
            // For now, ignore any branch span that was introduced by
            // expansion. This makes things like assert macros less noisy.
            if !raw_span.ctxt().outer_expn_data().is_root() {
                return None;
            }
            let span = unexpand_into_body_span(raw_span, hir_info.body_span)?;

            let bcb_from_marker =
                |marker: BlockMarkerId| basic_coverage_blocks.bcb_from_bb(block_markers[marker]?);

            let true_bcb = bcb_from_marker(true_marker)?;
            let false_bcb = bcb_from_marker(false_marker)?;

            Some(BranchPair { span, true_bcb, false_bcb })
        })
        .collect::<Vec<_>>()
}

pub(super) fn extract_mcdc_mappings(
    mir_body: &mir::Body<'_>,
    body_span: Span,
    basic_coverage_blocks: &CoverageGraph,
    mcdc_bitmap_bytes: &mut u32,
    mcdc_branches: &mut impl Extend<MCDCBranch>,
    mcdc_decisions: &mut impl Extend<MCDCDecision>,
) {
    let Some(coverage_info_hi) = mir_body.coverage_info_hi.as_deref() else { return };

    let block_markers = resolve_block_markers(coverage_info_hi, mir_body);

    let bcb_from_marker =
        |marker: BlockMarkerId| basic_coverage_blocks.bcb_from_bb(block_markers[marker]?);

    let check_branch_bcb =
        |raw_span: Span, true_marker: BlockMarkerId, false_marker: BlockMarkerId| {
            // For now, ignore any branch span that was introduced by
            // expansion. This makes things like assert macros less noisy.
            if !raw_span.ctxt().outer_expn_data().is_root() {
                return None;
            }
            let span = unexpand_into_body_span(raw_span, body_span)?;

            let true_bcb = bcb_from_marker(true_marker)?;
            let false_bcb = bcb_from_marker(false_marker)?;
            Some((span, true_bcb, false_bcb))
        };

    mcdc_branches.extend(coverage_info_hi.mcdc_branch_spans.iter().filter_map(
        |&mir::coverage::MCDCBranchSpan {
             span: raw_span,
             condition_info,
             true_marker,
             false_marker,
             decision_depth,
         }| {
            let (span, true_bcb, false_bcb) =
                check_branch_bcb(raw_span, true_marker, false_marker)?;
            Some(MCDCBranch { span, true_bcb, false_bcb, condition_info, decision_depth })
        },
    ));

    mcdc_decisions.extend(coverage_info_hi.mcdc_decision_spans.iter().filter_map(
        |decision: &mir::coverage::MCDCDecisionSpan| {
            let span = unexpand_into_body_span(decision.span, body_span)?;

            let end_bcbs = decision
                .end_markers
                .iter()
                .map(|&marker| bcb_from_marker(marker))
                .collect::<Option<_>>()?;

            // Each decision containing N conditions needs 2^N bits of space in
            // the bitmap, rounded up to a whole number of bytes.
            // The decision's "bitmap index" points to its first byte in the bitmap.
            let bitmap_idx = *mcdc_bitmap_bytes;
            *mcdc_bitmap_bytes += (1_u32 << decision.num_conditions).div_ceil(8);

            Some(MCDCDecision {
                span,
                end_bcbs,
                bitmap_idx,
                num_conditions: decision.num_conditions as u16,
                decision_depth: decision.decision_depth,
            })
        },
    ));
}
