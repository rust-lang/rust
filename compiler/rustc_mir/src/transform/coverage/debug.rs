use super::graph::BasicCoverageBlocks;
use super::spans::CoverageSpan;

use crate::util::pretty;
use crate::util::spanview::{self, SpanViewable};

use rustc_middle::mir::{self, TerminatorKind};
use rustc_middle::ty::TyCtxt;

/// Generates the MIR pass `CoverageSpan`-specific spanview dump file.
pub(crate) fn dump_coverage_spanview(
    tcx: TyCtxt<'tcx>,
    mir_body: &mir::Body<'tcx>,
    basic_coverage_blocks: &BasicCoverageBlocks,
    pass_name: &str,
    coverage_spans: &Vec<CoverageSpan>,
) {
    let mir_source = mir_body.source;
    let def_id = mir_source.def_id();

    let span_viewables = span_viewables(tcx, mir_body, basic_coverage_blocks, &coverage_spans);
    let mut file = pretty::create_dump_file(tcx, "html", None, pass_name, &0, mir_source)
        .expect("Unexpected error creating MIR spanview HTML file");
    let crate_name = tcx.crate_name(def_id.krate);
    let item_name = tcx.def_path(def_id).to_filename_friendly_no_crate();
    let title = format!("{}.{} - Coverage Spans", crate_name, item_name);
    spanview::write_document(tcx, def_id, span_viewables, &title, &mut file)
        .expect("Unexpected IO error dumping coverage spans as HTML");
}

/// Converts the computed `BasicCoverageBlock`s into `SpanViewable`s.
fn span_viewables(
    tcx: TyCtxt<'tcx>,
    mir_body: &mir::Body<'tcx>,
    basic_coverage_blocks: &BasicCoverageBlocks,
    coverage_spans: &Vec<CoverageSpan>,
) -> Vec<SpanViewable> {
    let mut span_viewables = Vec::new();
    for coverage_span in coverage_spans {
        let tooltip = coverage_span.format_coverage_statements(tcx, mir_body);
        let CoverageSpan { span, bcb_leader_bb: bb, .. } = coverage_span;
        let bcb = &basic_coverage_blocks[*bb];
        let id = bcb.id();
        let leader_bb = bcb.leader_bb();
        span_viewables.push(SpanViewable { bb: leader_bb, span: *span, id, tooltip });
    }
    span_viewables
}

/// Returns a simple string representation of a `TerminatorKind` variant, indenpendent of any
/// values it might hold.
pub(crate) fn term_type(kind: &TerminatorKind<'tcx>) -> &'static str {
    match kind {
        TerminatorKind::Goto { .. } => "Goto",
        TerminatorKind::SwitchInt { .. } => "SwitchInt",
        TerminatorKind::Resume => "Resume",
        TerminatorKind::Abort => "Abort",
        TerminatorKind::Return => "Return",
        TerminatorKind::Unreachable => "Unreachable",
        TerminatorKind::Drop { .. } => "Drop",
        TerminatorKind::DropAndReplace { .. } => "DropAndReplace",
        TerminatorKind::Call { .. } => "Call",
        TerminatorKind::Assert { .. } => "Assert",
        TerminatorKind::Yield { .. } => "Yield",
        TerminatorKind::GeneratorDrop => "GeneratorDrop",
        TerminatorKind::FalseEdge { .. } => "FalseEdge",
        TerminatorKind::FalseUnwind { .. } => "FalseUnwind",
        TerminatorKind::InlineAsm { .. } => "InlineAsm",
    }
}
