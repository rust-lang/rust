use rustc_index::IndexVec;
use rustc_middle::mir::coverage::{
    BlockMarkerId, BranchSpan, CoverageInfoHi, CoverageKind, Mapping, MappingKind,
};
use rustc_middle::mir::{self, BasicBlock, StatementKind};
use rustc_middle::ty::TyCtxt;

use crate::coverage::graph::CoverageGraph;
use crate::coverage::hir_info::ExtractedHirInfo;
use crate::coverage::spans::extract_refined_covspans;
use crate::coverage::unexpand::unexpand_into_body_span;

#[derive(Default)]
pub(crate) struct ExtractedMappings {
    pub(crate) mappings: Vec<Mapping>,
}

/// Extracts coverage-relevant spans from MIR, and uses them to create
/// coverage mapping data for inclusion in MIR.
pub(crate) fn extract_mappings_from_mir<'tcx>(
    tcx: TyCtxt<'tcx>,
    mir_body: &mir::Body<'tcx>,
    hir_info: &ExtractedHirInfo,
    graph: &CoverageGraph,
) -> ExtractedMappings {
    let mut mappings = vec![];

    // Extract ordinary code mappings from MIR statement/terminator spans.
    extract_refined_covspans(tcx, mir_body, hir_info, graph, &mut mappings);

    extract_branch_mappings(mir_body, hir_info, graph, &mut mappings);

    ExtractedMappings { mappings }
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

pub(super) fn extract_branch_mappings(
    mir_body: &mir::Body<'_>,
    hir_info: &ExtractedHirInfo,
    graph: &CoverageGraph,
    mappings: &mut Vec<Mapping>,
) {
    let Some(coverage_info_hi) = mir_body.coverage_info_hi.as_deref() else { return };

    let block_markers = resolve_block_markers(coverage_info_hi, mir_body);

    mappings.extend(coverage_info_hi.branch_spans.iter().filter_map(
        |&BranchSpan { span: raw_span, true_marker, false_marker }| try {
            // For now, ignore any branch span that was introduced by
            // expansion. This makes things like assert macros less noisy.
            if !raw_span.ctxt().outer_expn_data().is_root() {
                return None;
            }
            let span = unexpand_into_body_span(raw_span, hir_info.body_span)?;

            let bcb_from_marker = |marker: BlockMarkerId| graph.bcb_from_bb(block_markers[marker]?);

            let true_bcb = bcb_from_marker(true_marker)?;
            let false_bcb = bcb_from_marker(false_marker)?;

            Mapping { span, kind: MappingKind::Branch { true_bcb, false_bcb } }
        },
    ));
}
