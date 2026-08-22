//! MC/DC instrumentation enables MC/DC coverage to be performed on the
//! program. That is, for each "decision" (a boolean expression with at least
//! one logical operator), ensure that each "condition" (single operands of
//! the expression) is independently proven to have an effect on the outcome of
//! the decision. To do that, we need to :
//! 1. Detect decisions and their conditions in the AST (THIR).
//!
//! 2. Instrument the BDD (Binary Decision Diagram) (subset of MIR
//!     corresponding to the decision). The goal of this step is to make sure
//!     that upon finishing the evaluation of a decision, the program saves a
//!     trace of the test vector (i.e. the set of input values) that was just
//!     given. There are 3 requirements to do this:
//!     - Allocate a temporary variable during the evaluation of the
//!         decision, which should represent the current test vector being
//!         evaluated.
//!     - Upon ending a decision, save the aforementioned temporary variable
//!         somehow, so we know this input was exercised.
//!     - When evaluating a condition within the decision, update the
//!         aforementioned temporary variable to keep track of this specific
//!         condition evaluation.
//!     *Note* that it is necessary that all possible test vectors produce a
//!     unique value.
//!
//! 3. Adapt the codegen. Once all the processing is done pass all the data
//!     the backend needs to generate MC/DC entries.
//!     For LLVM, that includes providing it with "mappings", that are needed
//!     by the instrumentation backend, so the post-execution script can map
//!     back the coverage data to actual places in the source code, and calls
//!     to the `llvm.instrprof.mcdc.tvbitmapupdate` intrinsic were decisions
//!     end.
//!
//! The implementation is detailed below :
//!
//! 1. Detection of boolean decisions and conditions. This phase happens in
//!     `rustc_mir_build::builder::coverageinfo::mcdc` and it does 2 things.
//!     First, it detects boolean expressions, keeps a representation of all of
//!     them ([`rustc_middle::mir::coverage::mcdc::DecisionSpan`] and
//!     [`rustc_middle::mir::coverage::mcdc::ConditionSpan`]) along with the
//!     corresponding spans in the source code, and stores it in
//!     [`rustc_middle::mir::coverage::CoverageInfoHi`].
//!     Additionally, it alters the MIR building and inserts [`BlockMarkerId`]s
//!     to keep track of which basic blocks are generated from this decisions.
//!
//! 2. Instrumentation of MIR. This phase happens mostly in this module.
//!     First, [`extract_mcdc_mappings`] processes
//!     [`rustc_middle::mir::coverage::mcdc::DecisionSpan`]s and
//!     [`rustc_middle::mir::coverage::mcdc::ConditionSpan`]s to create
//!     "meta-mappings", which are essentially the structures we will give to
//!     LLVM but with some extra data we cannot get rid of just yet.
//!     [`extract_mcdc_mappings`] is also responsible for computing the index
//!     increment values of each condition outcome via
//!     [`calc_test_vectors_index`].
//!     > **Warning**: [`calc_test_vectors_index`] reimplements an algorithm
//!         from LLVM and should be kept synced with it, otherwise, test
//!         vectors may be wrongly indexed between compilation and LLVM
//!         reporting.
//!
//!     Once index increments are computed, we inject
//!     [`CoverageKind`]`::TmpIdxUpdate` and
//!     [`CoverageKind`]`::TvBitmapUpdate` instructions in MIR, and generate
//!     the [`Mapping`]s we'll be passing to LLVM.
//!

use std::collections::BTreeSet;

use rustc_data_structures::fx::FxIndexMap;
use rustc_index::IndexVec;
use rustc_middle::mir;
use rustc_middle::mir::coverage::mcdc::{ConditionId, ConditionInfo, ConditionSpan};
use rustc_middle::mir::coverage::{
    BasicCoverageBlock, BlockMarkerId, CoverageKind, FunctionMCDCExtraInfo, Mapping, MappingKind,
};
use rustc_span::{ExpnKind, Span};

use crate::coverage::expansion::ExpnTree;
use crate::coverage::graph::CoverageGraph;
use crate::coverage::hir_info::ExtractedHirInfo;
use crate::coverage::inject_statement;
use crate::coverage::mappings::resolve_block_markers;

pub(crate) type MCDCMetaMapping = (MCDCMetaDecisionMapping, Vec<MCDCMetaConditionMapping>);

/// MC/DC Condition Mapping with extra data to compute `num_test_vectors`.
#[derive(Debug)]
pub(crate) struct MCDCMetaConditionMapping {
    span: Span,
    condition_info: ConditionInfo,
    true_bcb: BasicCoverageBlock,
    false_bcb: BasicCoverageBlock,
    // Offset added to test vector idx if this branch is evaluated to true.
    true_incr: usize,
    // Offset added to test vector idx if this branch is evaluated to false.
    false_incr: usize,
}

impl MCDCMetaConditionMapping {
    fn new(
        span: Span,
        condition_info: ConditionInfo,
        true_bcb: BasicCoverageBlock,
        false_bcb: BasicCoverageBlock,
    ) -> Self {
        Self {
            span,
            condition_info,
            true_bcb,
            false_bcb,
            true_incr: usize::MAX,
            false_incr: usize::MAX,
        }
    }
}

impl From<MCDCMetaConditionMapping> for Mapping {
    fn from(
        MCDCMetaConditionMapping {
            span,
            condition_info,
            true_bcb,
            false_bcb,
            ..
        }: MCDCMetaConditionMapping,
    ) -> Self {
        Self {
            kind: MappingKind::MCDCCondition { true_bcb, false_bcb, mcdc_mappings: condition_info },
            span,
        }
    }
}

/// Holds all the information about an MC/DC decision mapping.
/// Later transformed into a regular LLVM-equivalent mapping.
#[derive(Debug)]
pub(crate) struct MCDCMetaDecisionMapping {
    span: Span,
    /// Output basic blocks where the executed test vector should be saved.
    end_bcbs: BTreeSet<BasicCoverageBlock>,
    /// STARTING index of the decision in the function's MCDC bitmap.
    bitmap_idx: u32,
    num_conditions: u16,
    num_test_vectors: usize,
    decision_depth: u16,
}

impl From<MCDCMetaDecisionMapping> for Mapping {
    fn from(
        MCDCMetaDecisionMapping { span, bitmap_idx, num_conditions, num_test_vectors, .. }: MCDCMetaDecisionMapping,
    ) -> Self {
        // LLVM expects the ENDING bitmap index.
        let bitmap_idx = bitmap_idx + num_test_vectors as u32;
        Self { kind: MappingKind::MCDCDecision { bitmap_idx, num_conditions }, span }
    }
}

#[derive(Default)]
struct BitmapIndexGen {
    idx: usize,
}

impl BitmapIndexGen {
    #[inline(always)]
    fn next(&mut self, num_tv: usize) -> Option<usize> {
        let ret = self.idx;
        let next_idx = ret + num_tv;

        if next_idx < MCDC_MAX_BITMAP_SIZE {
            self.idx = next_idx;
            Some(ret)
        } else {
            None
        }
    }

    /// Returns the total number of bytes needed to accommodate for all the test
    /// vectors of the function.
    #[inline(always)]
    fn total(self) -> usize {
        self.idx
    }
}

// LLVM uses `i32` to index the bitmap. Thus `i32::MAX` is the hard limit for
// number of all test vectors in a function.
const MCDC_MAX_BITMAP_SIZE: usize = i32::MAX as _;

pub(crate) fn extract_mcdc_mappings(
    mir_body: &mir::Body<'_>,
    hir_info: &ExtractedHirInfo,
    graph: &CoverageGraph,
    expn_tree: &ExpnTree,
) -> Option<(Vec<MCDCMetaMapping>, FunctionMCDCExtraInfo)> {
    let Some(coverage_info_hi) = mir_body.coverage_info_hi.as_deref() else { return None };
    let block_markers = resolve_block_markers(coverage_info_hi, mir_body);

    // For now, ignore any MCDC decision span that was introduced by
    // expansion. This makes things like assert macros less noisy.
    // FIXME(peron): Investigate better expansion support.
    let Some(node) = expn_tree.get(hir_info.body_span.ctxt()) else { return None };
    if node.expn_kind != ExpnKind::Root {
        return None;
    }

    // No decision to instrument, either because the function has no decision,
    // or we didn't ask for MC/DC instrumentation.
    if node.mcdc_spans.is_empty() {
        return None;
    }

    let bcb_from_marker = |marker: BlockMarkerId| graph.bcb_from_bb(block_markers[marker]?);

    let mut mcdc_mappings = vec![];
    let mut bitmap_idx_gen = BitmapIndexGen::default();
    let mut max_depth = 0;

    for (decision, conditions) in node.mcdc_spans.iter() {
        let end_bcbs = decision
            .end_markers
            .iter()
            .map(|&marker| bcb_from_marker(marker))
            .collect::<Option<BTreeSet<_>>>()?; // FIXME(renjisann): deal with bcb_from_marker errors.

        let mut condition_meta_mappings = vec![];

        for ConditionSpan { span, condition_info, true_marker, false_marker } in conditions {
            let true_bcb = bcb_from_marker(*true_marker)?;
            let false_bcb = bcb_from_marker(*false_marker)?;

            condition_meta_mappings.push(MCDCMetaConditionMapping::new(
                *span,
                *condition_info,
                true_bcb,
                false_bcb,
            ));
        }

        let num_test_vectors = calc_test_vectors_index(&mut condition_meta_mappings);
        let Some(bitmap_idx) = bitmap_idx_gen.next(num_test_vectors) else {
            tracing::debug!("Exceeded test vector limit");
            return None;
        };

        let decision_meta_mapping = MCDCMetaDecisionMapping {
            span: decision.span,
            end_bcbs,
            bitmap_idx: bitmap_idx as _,
            num_conditions: decision.num_conditions as _,
            num_test_vectors,
            decision_depth: decision.decision_depth,
        };

        max_depth = max_depth.max(decision.decision_depth);

        mcdc_mappings.push((decision_meta_mapping, condition_meta_mappings));
    }

    Some((
        mcdc_mappings,
        FunctionMCDCExtraInfo {
            bitmap_bits: bitmap_idx_gen.total(),
            num_temporaries: (max_depth as usize).saturating_add(1),
        },
    ))
}

pub(crate) fn inject_statements_for_decisions(
    mir_body: &mut mir::Body<'_>,
    graph: &CoverageGraph,
    mappings: &Vec<(MCDCMetaDecisionMapping, Vec<MCDCMetaConditionMapping>)>,
) {
    for (decision, conditions) in mappings {
        let &MCDCMetaDecisionMapping { ref end_bcbs, bitmap_idx, decision_depth, .. } = decision;

        // Insert llvm.instrprof.mcdc.tvbitmap.update values.
        for bcb in end_bcbs {
            let leader_bb = graph[*bcb].leader_bb();
            inject_statement(
                mir_body,
                CoverageKind::MCDCTestVectorBitmapUpdate { bitmap_idx, decision_depth },
                leader_bb,
            );
        }

        // Insert temporary variable updates for each visited condition.
        let tmp_indexes_increments = conditions.iter().flat_map(
            |MCDCMetaConditionMapping { true_bcb, false_bcb, true_incr, false_incr, .. }| {
                [(true_bcb, true_incr), (false_bcb, false_incr)]
            },
        );

        for (&bcb, &incr) in tmp_indexes_increments {
            let leader_bb = graph[bcb].leader_bb();
            inject_statement(
                mir_body,
                CoverageKind::MCDCTmpIdxUpdate { incr: incr as u32, decision_depth },
                leader_bb,
            );
        }
    }
}

// LLVM checks the executed test vector by accumulating indices of tested branches.
// We calculate number of all possible test vectors of the decision and assign indices
// to branches here.
// See [the rfc](https://discourse.llvm.org/t/rfc-coverage-new-algorithm-and-file-format-for-mc-dc/76798/)
// for more details about the algorithm.
// This function is mostly like [`TVIdxBuilder::TvIdxBuilder`](https://github.com/llvm/llvm-project/blob/d594d9f7f4dc6eb748b3261917db689fdc348b96/llvm/lib/ProfileData/Coverage/CoverageMapping.cpp#L226)
fn calc_test_vectors_index(conditions: &mut Vec<MCDCMetaConditionMapping>) -> usize {
    let mut indegree_stats = IndexVec::<ConditionId, usize>::from_elem_n(0, conditions.len());

    // `num_paths` is `width` described at the llvm rfc, which indicates how many paths reaching the condition node.
    let mut num_paths_stats = IndexVec::<ConditionId, usize>::from_elem_n(0, conditions.len());
    num_paths_stats[ConditionId::START] = 1;

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

    let mut decision_end_nodes = Vec::new();

    while let Some(branch) = queue.pop_front() {
        let ConditionInfo { condition_id, true_next_id, false_next_id }: ConditionInfo =
            branch.condition_info;

        let (false_incr, true_incr) = (&mut branch.false_incr, &mut branch.true_incr);

        // `width` of this branch
        let this_paths_count = num_paths_stats[condition_id];

        // Note. First check the false next to ensure conditions are touched in same order with llvm-cov.
        for (next, incr) in [(false_next_id, false_incr), (true_next_id, true_incr)] {
            if let Some(next_id) = next {
                let next_paths_count = &mut num_paths_stats[next_id];
                *incr = *next_paths_count;
                *next_paths_count = next_paths_count.saturating_add(this_paths_count);
                let next_indegree = &mut indegree_stats[next_id];
                *next_indegree -= 1;
                if *next_indegree == 0 {
                    queue.push_back(next_conditions.swap_remove(&next_id).expect(
                        "conditions with non-zero indegree before must be in next_conditions",
                    ));
                }
            } else {
                decision_end_nodes.push((this_paths_count, condition_id, incr));
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
