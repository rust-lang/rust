use std::collections::BTreeSet;

use rustc_data_structures::fx::FxIndexMap;
use rustc_index::IndexVec;
use rustc_middle::mir::coverage::{
    BlockMarkerId, BranchSpan, ConditionId, ConditionInfo, CoverageInfoHi, CoverageKind,
};
use rustc_middle::mir::{self, BasicBlock, StatementKind};
use rustc_middle::ty::TyCtxt;
use rustc_span::Span;

use crate::coverage::ExtractedHirInfo;
use crate::coverage::graph::{BasicCoverageBlock, CoverageGraph, START_BCB};
use crate::coverage::spans::extract_refined_covspans;
use crate::coverage::unexpand::unexpand_into_body_span;
use crate::errors::MCDCExceedsTestVectorLimit;

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
    pub(super) condition_info: ConditionInfo,
    // Offset added to test vector idx if this branch is evaluated to true.
    pub(super) true_index: usize,
    // Offset added to test vector idx if this branch is evaluated to false.
    pub(super) false_index: usize,
}

/// Associates an MC/DC decision with its join BCBs.
#[derive(Debug)]
pub(super) struct MCDCDecision {
    pub(super) span: Span,
    pub(super) end_bcbs: BTreeSet<BasicCoverageBlock>,
    pub(super) bitmap_idx: usize,
    pub(super) num_test_vectors: usize,
    pub(super) decision_depth: u16,
}

// LLVM uses `i32` to index the bitmap. Thus `i32::MAX` is the hard limit for number of all test vectors
// in a function.
const MCDC_MAX_BITMAP_SIZE: usize = i32::MAX as usize;

#[derive(Default)]
pub(super) struct ExtractedMappings {
    pub(super) code_mappings: Vec<CodeMapping>,
    pub(super) branch_pairs: Vec<BranchPair>,
    pub(super) mcdc_bitmap_bits: usize,
    pub(super) mcdc_degraded_branches: Vec<MCDCBranch>,
    pub(super) mcdc_mappings: Vec<(MCDCDecision, Vec<MCDCBranch>)>,
}

/// Extracts coverage-relevant spans from MIR, and associates them with
/// their corresponding BCBs.
pub(super) fn extract_all_mapping_info_from_mir<'tcx>(
    tcx: TyCtxt<'tcx>,
    mir_body: &mir::Body<'tcx>,
    hir_info: &ExtractedHirInfo,
    graph: &CoverageGraph,
) -> ExtractedMappings {
    let mut code_mappings = vec![];
    let mut branch_pairs = vec![];
    let mut mcdc_bitmap_bits = 0;
    let mut mcdc_degraded_branches = vec![];
    let mut mcdc_mappings = vec![];

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
        extract_refined_covspans(tcx, mir_body, hir_info, graph, &mut code_mappings);
    }

    branch_pairs.extend(extract_branch_pairs(mir_body, hir_info, graph));

    extract_mcdc_mappings(
        mir_body,
        tcx,
        hir_info.body_span,
        graph,
        &mut mcdc_bitmap_bits,
        &mut mcdc_degraded_branches,
        &mut mcdc_mappings,
    );

    ExtractedMappings {
        code_mappings,
        branch_pairs,
        mcdc_bitmap_bits,
        mcdc_degraded_branches,
        mcdc_mappings,
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
    graph: &CoverageGraph,
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

            let bcb_from_marker = |marker: BlockMarkerId| graph.bcb_from_bb(block_markers[marker]?);

            let true_bcb = bcb_from_marker(true_marker)?;
            let false_bcb = bcb_from_marker(false_marker)?;

            Some(BranchPair { span, true_bcb, false_bcb })
        })
        .collect::<Vec<_>>()
}

pub(super) fn extract_mcdc_mappings(
    mir_body: &mir::Body<'_>,
    tcx: TyCtxt<'_>,
    body_span: Span,
    graph: &CoverageGraph,
    mcdc_bitmap_bits: &mut usize,
    mcdc_degraded_branches: &mut impl Extend<MCDCBranch>,
    mcdc_mappings: &mut impl Extend<(MCDCDecision, Vec<MCDCBranch>)>,
) {
    let Some(coverage_info_hi) = mir_body.coverage_info_hi.as_deref() else { return };

    let block_markers = resolve_block_markers(coverage_info_hi, mir_body);

    let bcb_from_marker = |marker: BlockMarkerId| graph.bcb_from_bb(block_markers[marker]?);

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

    let to_mcdc_branch = |&mir::coverage::MCDCBranchSpan {
                              span: raw_span,
                              condition_info,
                              true_marker,
                              false_marker,
                          }| {
        let (span, true_bcb, false_bcb) = check_branch_bcb(raw_span, true_marker, false_marker)?;
        Some(MCDCBranch {
            span,
            true_bcb,
            false_bcb,
            condition_info,
            true_index: usize::MAX,
            false_index: usize::MAX,
        })
    };

    let mut get_bitmap_idx = |num_test_vectors: usize| -> Option<usize> {
        let bitmap_idx = *mcdc_bitmap_bits;
        let next_bitmap_bits = bitmap_idx.saturating_add(num_test_vectors);
        (next_bitmap_bits <= MCDC_MAX_BITMAP_SIZE).then(|| {
            *mcdc_bitmap_bits = next_bitmap_bits;
            bitmap_idx
        })
    };
    mcdc_degraded_branches
        .extend(coverage_info_hi.mcdc_degraded_branch_spans.iter().filter_map(to_mcdc_branch));

    mcdc_mappings.extend(coverage_info_hi.mcdc_spans.iter().filter_map(|(decision, branches)| {
        if branches.len() == 0 {
            return None;
        }
        let decision_span = unexpand_into_body_span(decision.span, body_span)?;

        let end_bcbs = decision
            .end_markers
            .iter()
            .map(|&marker| bcb_from_marker(marker))
            .collect::<Option<_>>()?;
        let mut branch_mappings: Vec<_> = branches.into_iter().filter_map(to_mcdc_branch).collect();
        if branch_mappings.len() != branches.len() {
            mcdc_degraded_branches.extend(branch_mappings);
            return None;
        }
        let num_test_vectors = calc_test_vectors_index(&mut branch_mappings);
        let Some(bitmap_idx) = get_bitmap_idx(num_test_vectors) else {
            tcx.dcx().emit_warn(MCDCExceedsTestVectorLimit {
                span: decision_span,
                max_num_test_vectors: MCDC_MAX_BITMAP_SIZE,
            });
            mcdc_degraded_branches.extend(branch_mappings);
            return None;
        };
        // LLVM requires span of the decision contains all spans of its conditions.
        // Usually the decision span meets the requirement well but in cases like macros it may not.
        let span = branch_mappings
            .iter()
            .map(|branch| branch.span)
            .reduce(|lhs, rhs| lhs.to(rhs))
            .map(
                |joint_span| {
                    if decision_span.contains(joint_span) { decision_span } else { joint_span }
                },
            )
            .expect("branch mappings are ensured to be non-empty as checked above");
        Some((
            MCDCDecision {
                span,
                end_bcbs,
                bitmap_idx,
                num_test_vectors,
                decision_depth: decision.decision_depth,
            },
            branch_mappings,
        ))
    }));
}

// LLVM checks the executed test vector by accumulating indices of tested branches.
// We calculate number of all possible test vectors of the decision and assign indices
// to branches here.
// See [the rfc](https://discourse.llvm.org/t/rfc-coverage-new-algorithm-and-file-format-for-mc-dc/76798/)
// for more details about the algorithm.
// This function is mostly like [`TVIdxBuilder::TvIdxBuilder`](https://github.com/llvm/llvm-project/blob/d594d9f7f4dc6eb748b3261917db689fdc348b96/llvm/lib/ProfileData/Coverage/CoverageMapping.cpp#L226)
fn calc_test_vectors_index(conditions: &mut Vec<MCDCBranch>) -> usize {
    let mut indegree_stats = IndexVec::<ConditionId, usize>::from_elem_n(0, conditions.len());
    // `num_paths` is `width` described at the llvm rfc, which indicates how many paths reaching the condition node.
    let mut num_paths_stats = IndexVec::<ConditionId, usize>::from_elem_n(0, conditions.len());
    let mut next_conditions = conditions
        .iter_mut()
        .map(|branch| {
            let ConditionInfo { condition_id, true_next_id, false_next_id } = branch.condition_info;
            [true_next_id, false_next_id]
                .into_iter()
                .flatten()
                .for_each(|next_id| indegree_stats[next_id] += 1);
            (condition_id, branch)
        })
        .collect::<FxIndexMap<_, _>>();

    let mut queue =
        std::collections::VecDeque::from_iter(next_conditions.swap_remove(&ConditionId::START));
    num_paths_stats[ConditionId::START] = 1;
    let mut decision_end_nodes = Vec::new();
    while let Some(branch) = queue.pop_front() {
        let ConditionInfo { condition_id, true_next_id, false_next_id } = branch.condition_info;
        let (false_index, true_index) = (&mut branch.false_index, &mut branch.true_index);
        let this_paths_count = num_paths_stats[condition_id];
        // Note. First check the false next to ensure conditions are touched in same order with llvm-cov.
        for (next, index) in [(false_next_id, false_index), (true_next_id, true_index)] {
            if let Some(next_id) = next {
                let next_paths_count = &mut num_paths_stats[next_id];
                *index = *next_paths_count;
                *next_paths_count = next_paths_count.saturating_add(this_paths_count);
                let next_indegree = &mut indegree_stats[next_id];
                *next_indegree -= 1;
                if *next_indegree == 0 {
                    queue.push_back(next_conditions.swap_remove(&next_id).expect(
                        "conditions with non-zero indegree before must be in next_conditions",
                    ));
                }
            } else {
                decision_end_nodes.push((this_paths_count, condition_id, index));
            }
        }
    }
    assert!(next_conditions.is_empty(), "the decision tree has untouched nodes");
    let mut cur_idx = 0;
    // LLVM hopes the end nodes are sorted in descending order by `num_paths` so that it can
    // optimize bitmap size for decisions in tree form such as `a && b && c && d && ...`.
    decision_end_nodes.sort_by_key(|(num_paths, _, _)| usize::MAX - *num_paths);
    for (num_paths, condition_id, index) in decision_end_nodes {
        assert_eq!(
            num_paths, num_paths_stats[condition_id],
            "end nodes should not be updated since they were visited"
        );
        assert_eq!(*index, usize::MAX, "end nodes should not be assigned index before");
        *index = cur_idx;
        cur_idx += num_paths;
    }
    cur_idx
}
