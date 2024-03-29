use rustc_data_structures::fx::FxHashMap;
use rustc_index::{bit_set::BitSet, IndexVec};
use rustc_middle::{
    mir::{
        self,
        coverage::{self, ConditionId, CoverageKind, DecisionId, DecisionMarkerId},
        BasicBlock, StatementKind,
    },
    ty::TyCtxt,
};

use crate::coverage::{graph::CoverageGraph, inject_statement};
use crate::errors::MCDCTooManyConditions;

const MAX_COND_DECISION: u32 = 6;

#[allow(dead_code)]

struct Decisions {
    data: IndexVec<DecisionId, DecisionData>,
    needed_bytes: u32,
}

impl Decisions {
    /// Use MIR markers and THIR extracted data to create the data we need for
    /// MCDC instrumentation.
    ///
    /// Note: MCDC Instrumentation might skip some decisions that contains too
    /// may conditions.
    pub fn extract_decisions<'tcx>(
        tcx: TyCtxt<'tcx>,
        branch_info: &coverage::BranchInfo,
        mir_body: &mir::Body<'_>,
    ) -> Self {
        let mut decisions_builder: IndexVec<DecisionId, DecisionDataBuilder> = IndexVec::new();
        let mut decm_id_to_dec_id: FxHashMap<DecisionMarkerId, DecisionId> = Default::default();
        let mut ignored_decisions: BitSet<DecisionMarkerId> = BitSet::new_empty(
            branch_info.decision_spans.last_index().map(|i| i.as_usize()).unwrap_or(0) + 1,
        );
        let mut needed_bytes: u32 = 0;

        // Start by gathering all the decisions.
        for (bb, data) in mir_body.basic_blocks.iter_enumerated() {
            for statement in &data.statements {
                match &statement.kind {
                    StatementKind::Coverage(CoverageKind::MCDCDecisionEntryMarker { decm_id }) => {
                        assert!(
                            !decm_id_to_dec_id.contains_key(decm_id)
                                && !ignored_decisions.contains(*decm_id),
                            "Duplicated decision marker id {:?}.",
                            decm_id
                        );

                        // Skip uninstrumentable conditions and flag them
                        // as ignored for the rest of the process.
                        let dec_span = &branch_info.decision_spans[*decm_id];
                        if dec_span.num_conditions > MAX_COND_DECISION {
                            tcx.dcx().emit_warn(MCDCTooManyConditions {
                                span: dec_span.span,
                                num_conditions: dec_span.num_conditions,
                                max_conditions: MAX_COND_DECISION,
                            });
                            ignored_decisions.insert(*decm_id);
                        } else {
                            decm_id_to_dec_id.insert(
                                *decm_id,
                                decisions_builder.push(DecisionDataBuilder::new(bb, needed_bytes)),
                            );
                            needed_bytes += 1 << dec_span.num_conditions;
                        }
                    }
                    _ => (),
                }
            }
        }

        for (bb, data) in mir_body.basic_blocks.iter_enumerated() {
            for statement in &data.statements {
                let StatementKind::Coverage(cov_kind) = &statement.kind else {
                    continue;
                };
                use CoverageKind::*;
                match cov_kind {
                    // Handled above
                    // MCDCDecisionEntryMarker { decm_id } => {
                    // }
                    MCDCDecisionOutputMarker { decm_id, outcome } => {
                        if let Some(decision_id) = decm_id_to_dec_id.get(decm_id) {
                            if *outcome {
                                decisions_builder[*decision_id].set_then_bb(bb);
                            } else {
                                decisions_builder[*decision_id].set_else_bb(bb);
                            }
                        } else if !ignored_decisions.contains(*decm_id) {
                            // If id is not in the mapping vector nor in the ignored IDs bitset,
                            // It means we have not encountered the corresponding DecisionEntryMarker.
                            bug!(
                                "Decision output marker {:?} coming before its decision entry marker.",
                                decm_id
                            );
                        }
                    }
                    MCDCConditionEntryMarker { decm_id, condm_id } => {
                        if let Some(decision_id) = decm_id_to_dec_id.get(decm_id) {
                            debug!("TODO MCDCConditionEntryMarker");
                            let dec_data = &mut decisions_builder[*decision_id];
                            dec_data.conditions.push(bb);
                        } else if !ignored_decisions.contains(*decm_id) {
                            // If id is not in the mapping vector nor in the ignored IDs bitset,
                            // It means we have not encountered the corresponding DecisionEntryMarker.
                            bug!(
                                "Condition marker {:?} references unknown decision entry marker.",
                                condm_id
                            );
                        }
                    }
                    MCDCConditionOutputMarker { decm_id, condm_id, outcome: _ } => {
                        if let Some(_decision_id) = decm_id_to_dec_id.get(decm_id) {
                            debug!("TODO MCDCConditionOutcomeMarker");
                        } else if !ignored_decisions.contains(*decm_id) {
                            // If id is not in the mapping vector nor in the ignored IDs bitset,
                            // It means we have not encountered the corresponding DecisionEntryMarker.
                            bug!(
                                "Condition marker {:?} references unknown decision entry marker.",
                                condm_id
                            );
                        }
                    }
                    _ => (), // Ignore other marker kinds.
                }
            }
        }

        Self {
            data: IndexVec::from_iter(decisions_builder.into_iter().map(|b| b.into_done())),
            needed_bytes,
        }
    }
}

// FIXME(dprn): Remove allow dead code
#[allow(unused)]
struct DecisionData {
    entry_bb: BasicBlock,

    /// Decision's offset in the global TV update table.
    bitmap_idx: u32,

    conditions: IndexVec<ConditionId, BasicBlock>,
    then_bb: BasicBlock,
    else_bb: BasicBlock,
}

// FIXME(dprn): Remove allow dead code
#[allow(dead_code)]
impl DecisionData {
    pub fn new(
        entry_bb: BasicBlock,
        bitmap_idx: u32,
        then_bb: BasicBlock,
        else_bb: BasicBlock,
    ) -> Self {
        Self { entry_bb, bitmap_idx, conditions: IndexVec::new(), then_bb, else_bb }
    }
}

struct DecisionDataBuilder {
    entry_bb: BasicBlock,

    /// Decision's offset in the global TV update table.
    bitmap_idx: u32,

    conditions: IndexVec<ConditionId, BasicBlock>,
    then_bb: Option<BasicBlock>,
    else_bb: Option<BasicBlock>,
}

// FIXME(dprn): Remove allow dead code
#[allow(dead_code)]
impl DecisionDataBuilder {
    pub fn new(entry_bb: BasicBlock, bitmap_idx: u32) -> Self {
        Self {
            entry_bb,
            bitmap_idx,
            conditions: Default::default(),
            then_bb: Default::default(),
            else_bb: Default::default(),
        }
    }

    pub fn set_then_bb(&mut self, then_bb: BasicBlock) -> &mut Self {
        self.then_bb = Some(then_bb);
        self
    }

    pub fn set_else_bb(&mut self, else_bb: BasicBlock) -> &mut Self {
        self.else_bb = Some(else_bb);
        self
    }

    pub fn into_done(self) -> DecisionData {
        assert!(!self.conditions.is_empty(), "Empty condition vector");

        DecisionData {
            entry_bb: self.entry_bb,
            bitmap_idx: self.bitmap_idx,
            conditions: self.conditions,
            then_bb: self.then_bb.expect("Missing then_bb"),
            else_bb: self.else_bb.expect("Missing else_bb"),
        }
    }
}

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
    let _decision_data = Decisions::extract_decisions(tcx, branch_info, mir_body);

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
    inject_statement(mir_body, CoverageKind::MCDCBitmapRequire { needed_bytes }, mir::START_BLOCK);

    // For each decision:
    // - Find its 'root' basic block
    // - Insert a 'reset CDM' instruction
    // - for each branch, find the root condition, give it an index and
    //   call a condbitmapUpdate on it
    // - Get the Output markers, and insert goto blocks before to put a
    //   tvbitmapupdate on it.
}
