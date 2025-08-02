use rustc_middle::mir::coverage::{
    CoverageKind, DecisionInfo, FunctionCoverageInfo, Mapping, MappingKind,
};
use rustc_middle::mir::{self, BasicBlock, Statement, StatementKind, TerminatorKind};
use rustc_middle::ty::TyCtxt;
use tracing::{debug, debug_span, trace};

use crate::coverage::counters::BcbCountersData;
use crate::coverage::graph::CoverageGraph;
use crate::coverage::mappings::ExtractedMappings;

mod counters;
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
    let extracted_mappings =
        mappings::extract_all_mapping_info_from_mir(tcx, mir_body, &hir_info, &graph);

    let mappings = create_mappings(&extracted_mappings);
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
    inject_mcdc_statements(mir_body, &graph, &extracted_mappings);

    let mcdc_num_condition_bitmaps = extracted_mappings
        .mcdc_mappings
        .iter()
        .map(|&(mappings::MCDCDecision { decision_depth, .. }, _)| decision_depth)
        .max()
        .map_or(0, |max| usize::from(max) + 1);

    mir_body.function_coverage_info = Some(Box::new(FunctionCoverageInfo {
        function_source_hash: hir_info.function_source_hash,

        node_flow_data,
        priority_list,

        mappings,

        mcdc_bitmap_bits: extracted_mappings.mcdc_bitmap_bits,
        mcdc_num_condition_bitmaps,
    }));
}

/// For each coverage span extracted from MIR, create a corresponding mapping.
///
/// FIXME(Zalathar): This used to be where BCBs in the extracted mappings were
/// resolved to a `CovTerm`. But that is now handled elsewhere, so this
/// function can potentially be simplified even further.
fn create_mappings(extracted_mappings: &ExtractedMappings) -> Vec<Mapping> {
    // Fully destructure the mappings struct to make sure we don't miss any kinds.
    let ExtractedMappings {
        code_mappings,
        branch_pairs,
        mcdc_bitmap_bits: _,
        mcdc_degraded_branches,
        mcdc_mappings,
    } = extracted_mappings;
    let mut mappings = Vec::new();

    mappings.extend(code_mappings.iter().map(
        // Ordinary code mappings are the simplest kind.
        |&mappings::CodeMapping { span, bcb }| {
            let kind = MappingKind::Code { bcb };
            Mapping { kind, span }
        },
    ));

    mappings.extend(branch_pairs.iter().map(
        |&mappings::BranchPair { span, true_bcb, false_bcb }| {
            let kind = MappingKind::Branch { true_bcb, false_bcb };
            Mapping { kind, span }
        },
    ));

    // MCDC branch mappings are appended with their decisions in case decisions were ignored.
    mappings.extend(mcdc_degraded_branches.iter().map(
        |&mappings::MCDCBranch {
             span,
             true_bcb,
             false_bcb,
             condition_info: _,
             true_index: _,
             false_index: _,
         }| { Mapping { kind: MappingKind::Branch { true_bcb, false_bcb }, span } },
    ));

    for (decision, branches) in mcdc_mappings {
        // FIXME(#134497): Previously it was possible for some of these branch
        // conversions to fail, in which case the remaining branches in the
        // decision would be degraded to plain `MappingKind::Branch`.
        // The changes in #134497 made that failure impossible, because the
        // fallible step was deferred to codegen. But the corresponding code
        // in codegen wasn't updated to detect the need for a degrade step.
        let conditions = branches
            .into_iter()
            .map(
                |&mappings::MCDCBranch {
                     span,
                     true_bcb,
                     false_bcb,
                     condition_info,
                     true_index: _,
                     false_index: _,
                 }| {
                    Mapping {
                        kind: MappingKind::MCDCBranch {
                            true_bcb,
                            false_bcb,
                            mcdc_params: condition_info,
                        },
                        span,
                    }
                },
            )
            .collect::<Vec<_>>();

        // LLVM requires end index for counter mapping regions.
        let kind = MappingKind::MCDCDecision(DecisionInfo {
            bitmap_idx: (decision.bitmap_idx + decision.num_test_vectors) as u32,
            num_conditions: u16::try_from(conditions.len()).unwrap(),
        });
        let span = decision.span;
        mappings.extend(std::iter::once(Mapping { kind, span }).chain(conditions.into_iter()));
    }

    mappings
}

/// Inject any necessary coverage statements into MIR, so that they influence codegen.
fn inject_coverage_statements<'tcx>(mir_body: &mut mir::Body<'tcx>, graph: &CoverageGraph) {
    for (bcb, data) in graph.iter_enumerated() {
        let target_bb = data.leader_bb();
        inject_statement(mir_body, CoverageKind::VirtualCounter { bcb }, target_bb);
    }
}

/// For each conditions inject statements to update condition bitmap after it has been evaluated.
/// For each decision inject statements to update test vector bitmap after it has been evaluated.
fn inject_mcdc_statements<'tcx>(
    mir_body: &mut mir::Body<'tcx>,
    graph: &CoverageGraph,
    extracted_mappings: &ExtractedMappings,
) {
    for (decision, conditions) in &extracted_mappings.mcdc_mappings {
        // Inject test vector update first because `inject_statement` always insert new statement at head.
        for &end in &decision.end_bcbs {
            let end_bb = graph[end].leader_bb();
            inject_statement(
                mir_body,
                CoverageKind::TestVectorBitmapUpdate {
                    bitmap_idx: decision.bitmap_idx as u32,
                    decision_depth: decision.decision_depth,
                },
                end_bb,
            );
        }

        for &mappings::MCDCBranch {
            span: _,
            true_bcb,
            false_bcb,
            condition_info: _,
            true_index,
            false_index,
        } in conditions
        {
            for (index, bcb) in [(false_index, false_bcb), (true_index, true_bcb)] {
                let bb = graph[bcb].leader_bb();
                inject_statement(
                    mir_body,
                    CoverageKind::CondBitmapUpdate {
                        index: index as u32,
                        decision_depth: decision.decision_depth,
                    },
                    bb,
                );
            }
        }
    }
}

fn inject_statement(mir_body: &mut mir::Body<'_>, counter_kind: CoverageKind, bb: BasicBlock) {
    debug!("  injecting statement {counter_kind:?} for {bb:?}");
    let data = &mut mir_body[bb];
    let source_info = data.terminator().source_info;
    let statement = Statement::new(source_info, StatementKind::Coverage(counter_kind));
    data.statements.insert(0, statement);
}
