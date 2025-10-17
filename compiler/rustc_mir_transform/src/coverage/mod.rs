use rustc_middle::mir::coverage::{CoverageKind, FunctionCoverageInfo};
use rustc_middle::mir::{self, BasicBlock, Statement, StatementKind, TerminatorKind};
use rustc_middle::ty::TyCtxt;
use tracing::{debug, debug_span, trace};

use crate::coverage::counters::BcbCountersData;
use crate::coverage::graph::CoverageGraph;
use crate::coverage::mappings::ExtractedMappings;

mod counters;
mod expansion;
mod graph;
mod hir_info;
mod mappings;
pub(super) mod query;
mod spans;
#[cfg(test)]
mod tests;
mod unexpand;

/// Inserts `StatementKind::Coverage` statements that either instrument the binary with injected
/// counters, via intrinsic `llvm.instrprof.increment`, and/or inject metadata used during codegen
/// to construct the coverage map.
pub(super) struct InstrumentCoverage;

impl<'tcx> crate::MirPass<'tcx> for InstrumentCoverage {
    fn is_enabled(&self, sess: &rustc_session::Session) -> bool {
        sess.instrument_coverage()
    }

    fn run_pass(&self, tcx: TyCtxt<'tcx>, mir_body: &mut mir::Body<'tcx>) {
        let mir_source = mir_body.source;

        // This pass runs after MIR promotion, but before promoted MIR starts to
        // be transformed, so it should never see promoted MIR.
        assert!(mir_source.promoted.is_none());

        let def_id = mir_source.def_id().expect_local();

        if !tcx.is_eligible_for_coverage(def_id) {
            trace!("InstrumentCoverage skipped for {def_id:?} (not eligible)");
            return;
        }

        // An otherwise-eligible function is still skipped if its start block
        // is known to be unreachable.
        match mir_body.basic_blocks[mir::START_BLOCK].terminator().kind {
            TerminatorKind::Unreachable => {
                trace!("InstrumentCoverage skipped for unreachable `START_BLOCK`");
                return;
            }
            _ => {}
        }

        instrument_function_for_coverage(tcx, mir_body);
    }

    fn is_required(&self) -> bool {
        false
    }
}

fn instrument_function_for_coverage<'tcx>(tcx: TyCtxt<'tcx>, mir_body: &mut mir::Body<'tcx>) {
    let def_id = mir_body.source.def_id();
    let _span = debug_span!("instrument_function_for_coverage", ?def_id).entered();

    let hir_info = hir_info::extract_hir_info(tcx, def_id.expect_local());

    // Build the coverage graph, which is a simplified view of the MIR control-flow
    // graph that ignores some details not relevant to coverage instrumentation.
    let graph = CoverageGraph::from_mir(mir_body);

    ////////////////////////////////////////////////////
    // Extract coverage spans and other mapping info from MIR.
    let ExtractedMappings { mappings } =
        mappings::extract_mappings_from_mir(tcx, mir_body, &hir_info, &graph);
    if mappings.is_empty() {
        // No spans could be converted into valid mappings, so skip this function.
        debug!("no spans could be converted into valid mappings; skipping");
        return;
    }

    // Use the coverage graph to prepare intermediate data that will eventually
    // be used to assign physical counters and counter expressions to points in
    // the control-flow graph.
    let BcbCountersData { node_flow_data, priority_list } =
        counters::prepare_bcb_counters_data(&graph);

    // Inject coverage statements into MIR.
    inject_coverage_statements(mir_body, &graph);

    mir_body.function_coverage_info = Some(Box::new(FunctionCoverageInfo {
        function_source_hash: hir_info.function_source_hash,

        node_flow_data,
        priority_list,

        mappings,
    }));
}

/// Inject any necessary coverage statements into MIR, so that they influence codegen.
fn inject_coverage_statements<'tcx>(mir_body: &mut mir::Body<'tcx>, graph: &CoverageGraph) {
    for (bcb, data) in graph.iter_enumerated() {
        let target_bb = data.leader_bb();
        inject_statement(mir_body, CoverageKind::VirtualCounter { bcb }, target_bb);
    }
}

fn inject_statement(mir_body: &mut mir::Body<'_>, counter_kind: CoverageKind, bb: BasicBlock) {
    debug!("  injecting statement {counter_kind:?} for {bb:?}");
    let data = &mut mir_body[bb];
    let source_info = data.terminator().source_info;
    let statement = Statement::new(source_info, StatementKind::Coverage(counter_kind));
    data.statements.insert(0, statement);
}
