use rustc_middle::{
    mir::{self, coverage::CoverageKind, Statement},
    ty::TyCtxt,
};

use crate::coverage::graph::CoverageGraph;
use crate::errors::MCDCTooManyConditions;

const MAX_COND_DECISION: u32 = 6;


/// If MCDC coverage is enabled, add MCDC instrumentation to the function.
///
/// Assume a decision to be the following:
///
///     if (A && B) || C { then(); } else { otherwise(); }
///
/// The corresponding BDD (Binary Decision Diagram) will look like so:
///
/// ```
///        │
///       ┌▼┐
///       │A│
///    ┌──┴─┴──┐
///    │t     f│
///    │       │
///   ┌▼┐     ┌▼┐
///   │B├─────►C├───┐
///   └┬┘  f  └┬┘  f│
///    │t      │t   │
/// ┌──▼───┐   │  ┌─▼─────────┐
/// │then()◄───┘  │otherwise()│
/// └──────┘      └───────────┘
/// ```
///
/// The coverage graph is similar, up to unwinding mechanics. The goal is to
/// instrument each edge of the BDD to update two bitmaps.
///
/// The first bitmap (CBM) is updated upon the evaluation of each contidion.
/// It tracks the state of a condition at a given instant.
/// is given an index, such that when the decision is taken, CBM represents the
/// state of all conditions that were evaluated (1 for true, 0 for
/// false/skipped).
///
/// The second bitmap (TVBM) is the test vector bitmap. It tracks all the test
/// vectors that were executed during the program's life. It is updated before
/// branching to `then()` or `otherwise()` by using CBM as an integer n and
/// setting the nth integer of TVBM to 1.
/// Basically, we do `TVBM |= 1 << CBM`.
///
/// Note: This technique is very sub-optimal, as it implies that TVBM to have a
/// size of 2^n bits, (n being the number of conditions in the decision) to
/// account for every combination, even though only a subset of theses
/// combinations are actually achievable because of logical operators
/// short-circuit semantics.
/// An improvement to this instrumentation is being implemented in Clang and
/// shall be ported to Rustc in the future:
/// https://discourse.llvm.org/t/rfc-coverage-new-algorithm-and-file-format-for-mc-dc/76798
///
/// In the meantime, to follow the original implementation of Clang, we use this
/// suboptimal technique, while limiting the size of the decisions we can
/// instrument to 6 conditions.
///
/// Will do the following things :
/// 1. Add an instruction in the first basic block for the codegen to call
///     the `instrprof.mcdc.parameters` instrinsic.
/// 2. Before each decision, add an instruction to reset CBM.
/// 3. Add an isntruction to update CBM upon condition evaluation.
/// 4. Add an instruction to update TVBM with the CBM value before jumping
///     to the `then` or `else` block.
/// 5. Build mappings for the reporting tools to get the results and transpose
///     it to the source code.
pub fn instrument_function_mcdc<'tcx>(
    tcx: TyCtxt<'tcx>,
    mir_body: &mut mir::Body<'tcx>,
    coverage_graph: &CoverageGraph,
) {
    let _ = coverage_graph;
    if !tcx.sess.instrument_coverage_mcdc() {
        return;
    }
    debug!("Called instrument_function_mcdc()");

    let Some(branch_info) = mir_body.coverage_branch_info.as_deref() else {
        return;
    };

    let mut needed_bytes = 0;
    let mut bitmap_indexes = vec![];

    for dec_span in branch_info.decision_spans.iter() {
        // Skip uninstrumentable conditions.
        if dec_span.num_conditions > MAX_COND_DECISION {
            tcx.dcx().emit_warn(MCDCTooManyConditions {
                span: dec_span.span,
                num_conditions: dec_span.num_conditions,
                max_conditions: MAX_COND_DECISION,
            });
            continue;
        }
        bitmap_indexes.push(needed_bytes);
        needed_bytes += 1 << dec_span.num_conditions;
    }

    if needed_bytes == 0 {
        // No decision to instrument
        return;
    }

    // In the first BB, specify that we need to allocate bitmaps.
    let entry_bb = &mut mir_body.basic_blocks_mut()[mir::START_BLOCK];
    let mcdc_parameters_statement = Statement {
        source_info: entry_bb.terminator().source_info,
        kind: mir::StatementKind::Coverage(CoverageKind::MCDCBitmapRequire { needed_bytes }),
    };
    entry_bb.statements.insert(0, mcdc_parameters_statement);

    // For each decision:
    // - Find its 'root' basic block
    // - Insert a 'reset CDM' instruction
    // - for each branch, find the root condition, give it an index and
    //   call a condbitmapUpdate on it
    // - Get the Output markers, and insert goto blocks before to put a
    //   tvbitmapupdate on it.
}
