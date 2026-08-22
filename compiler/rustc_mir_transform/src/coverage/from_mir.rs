use rustc_middle::mir::coverage::CoverageKind;
use rustc_middle::mir::{self, Statement, StatementKind};
use rustc_span::Span;

use crate::coverage::graph::{BasicCoverageBlock, CoverageGraph};
use crate::coverage::hir_info::ExtractedHirInfo;

#[derive(Debug)]
pub(crate) struct RawSpanFromMir {
    /// A span that has been extracted from a MIR marker statement, but
    /// hasn't been "unexpanded", so it might not lie within the function body
    /// span and might be part of an expansion with a different context.
    pub(crate) raw_span: Span,
    pub(crate) bcb: BasicCoverageBlock,
}

/// Generates an initial set of coverage spans from marker statements in the function's
/// MIR body, each associated with its corresponding node in the coverage graph.
///
/// FIXME(Zalathar): This extraction is currently in a transitional state, since we're
/// no longer trying to heuristically recover meaningful spans from MIR soup, but we
/// haven't yet fully embraced the possibilities of HIR-aware analysis.
pub(crate) fn extract_raw_spans_from_mir<'tcx>(
    mir_body: &mir::Body<'tcx>,
    hir_info: &ExtractedHirInfo,
    graph: &CoverageGraph,
) -> Vec<RawSpanFromMir> {
    let mut raw_spans = vec![];

    // We only care about blocks that are part of the coverage graph.
    for (bcb, bcb_data) in graph.iter_enumerated() {
        // A coverage graph node can consist of multiple basic blocks.
        for &bb in &bcb_data.basic_blocks {
            let statements = mir_body[bb].statements.iter();
            raw_spans.extend(
                statements
                    .filter_map(|stmt| filtered_statement_span(hir_info, stmt))
                    .map(|raw_span: Span| RawSpanFromMir { raw_span, bcb }),
            );
        }
    }

    raw_spans
}

/// If the MIR `Statement` has a span contributive to computing coverage spans,
/// return it; otherwise return `None`.
fn filtered_statement_span<'tcx>(
    hir_info: &ExtractedHirInfo,
    statement: &Statement<'tcx>,
) -> Option<Span> {
    let StatementKind::Coverage(CoverageKind::Point { point_kind: _, hir_id }) = statement.kind
    else {
        return None;
    };
    if hir_info.nodes_to_ignore.contains(&hir_id) {
        return None;
    }

    Some(statement.source_info.span)
}
