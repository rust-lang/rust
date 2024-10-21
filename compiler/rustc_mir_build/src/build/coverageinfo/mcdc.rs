use std::collections::VecDeque;

use rustc_middle::bug;
use rustc_middle::mir::coverage::{
    BlockMarkerId, ConditionId, ConditionInfo, MCDCBranchSpan, MCDCDecisionSpan,
};
use rustc_middle::mir::{BasicBlock, SourceInfo};
use rustc_middle::thir::LogicalOp;
use rustc_middle::ty::TyCtxt;
use rustc_span::Span;

use crate::build::Builder;
use crate::errors::MCDCExceedsConditionLimit;

/// LLVM uses `i16` to represent condition id. Hence `i16::MAX` is the hard limit for number of
/// conditions in a decision.
const MAX_CONDITIONS_IN_DECISION: usize = i16::MAX as usize;

#[derive(Default)]
struct MCDCDecisionCtx {
    /// To construct condition evaluation tree.
    decision_stack: VecDeque<ConditionInfo>,
    processing_decision: Option<MCDCDecisionSpan>,
    conditions: Vec<MCDCBranchSpan>,
}

struct MCDCState {
    decision_ctx_stack: Vec<MCDCDecisionCtx>,
}

impl MCDCState {
    fn new() -> Self {
        Self { decision_ctx_stack: vec![MCDCDecisionCtx::default()] }
    }

    /// Decision depth is given as a u16 to reduce the size of the `CoverageKind`,
    /// as it is very unlikely that the depth ever reaches 2^16.
    #[inline]
    fn decision_depth(&self) -> u16 {
        match u16::try_from(self.decision_ctx_stack.len())
            .expect(
                "decision depth did not fit in u16, this is likely to be an instrumentation error",
            )
            .checked_sub(1)
        {
            Some(d) => d,
            None => bug!("Unexpected empty decision stack"),
        }
    }

    // At first we assign ConditionIds for each sub expression.
    // If the sub expression is composite, re-assign its ConditionId to its LHS and generate a new ConditionId for its RHS.
    //
    // Example: "x = (A && B) || (C && D) || (D && F)"
    //
    //      Visit Depth1:
    //              (A && B) || (C && D) || (D && F)
    //              ^-------LHS--------^    ^-RHS--^
    //                      ID=1              ID=2
    //
    //      Visit LHS-Depth2:
    //              (A && B) || (C && D)
    //              ^-LHS--^    ^-RHS--^
    //                ID=1        ID=3
    //
    //      Visit LHS-Depth3:
    //               (A && B)
    //               LHS   RHS
    //               ID=1  ID=4
    //
    //      Visit RHS-Depth3:
    //                         (C && D)
    //                         LHS   RHS
    //                         ID=3  ID=5
    //
    //      Visit RHS-Depth2:              (D && F)
    //                                     LHS   RHS
    //                                     ID=2  ID=6
    //
    //      Visit Depth1:
    //              (A && B)  || (C && D)  || (D && F)
    //              ID=1  ID=4   ID=3  ID=5   ID=2  ID=6
    //
    // A node ID of '0' always means MC/DC isn't being tracked.
    //
    // If a "next" node ID is '0', it means it's the end of the test vector.
    //
    // As the compiler tracks expression in pre-order, we can ensure that condition info of parents are always properly assigned when their children are visited.
    // - If the op is AND, the "false_next" of LHS and RHS should be the parent's "false_next". While "true_next" of the LHS is the RHS, the "true next" of RHS is the parent's "true_next".
    // - If the op is OR, the "true_next" of LHS and RHS should be the parent's "true_next". While "false_next" of the LHS is the RHS, the "false next" of RHS is the parent's "false_next".
    fn record_conditions(&mut self, op: LogicalOp, span: Span) {
        let decision_depth = self.decision_depth();
        let Some(decision_ctx) = self.decision_ctx_stack.last_mut() else {
            bug!("Unexpected empty decision_ctx_stack")
        };
        let decision = match decision_ctx.processing_decision.as_mut() {
            Some(decision) => {
                decision.span = decision.span.to(span);
                decision
            }
            None => decision_ctx.processing_decision.insert(MCDCDecisionSpan {
                span,
                num_conditions: 0,
                end_markers: vec![],
                decision_depth,
            }),
        };

        let parent_condition = decision_ctx.decision_stack.pop_back().unwrap_or_else(|| {
            assert_eq!(
                decision.num_conditions, 0,
                "decision stack must be empty only for empty decision"
            );
            decision.num_conditions += 1;
            ConditionInfo {
                condition_id: ConditionId::START,
                true_next_id: None,
                false_next_id: None,
            }
        });
        let lhs_id = parent_condition.condition_id;

        let rhs_condition_id = ConditionId::from(decision.num_conditions);
        decision.num_conditions += 1;
        let (lhs, rhs) = match op {
            LogicalOp::And => {
                let lhs = ConditionInfo {
                    condition_id: lhs_id,
                    true_next_id: Some(rhs_condition_id),
                    false_next_id: parent_condition.false_next_id,
                };
                let rhs = ConditionInfo {
                    condition_id: rhs_condition_id,
                    true_next_id: parent_condition.true_next_id,
                    false_next_id: parent_condition.false_next_id,
                };
                (lhs, rhs)
            }
            LogicalOp::Or => {
                let lhs = ConditionInfo {
                    condition_id: lhs_id,
                    true_next_id: parent_condition.true_next_id,
                    false_next_id: Some(rhs_condition_id),
                };
                let rhs = ConditionInfo {
                    condition_id: rhs_condition_id,
                    true_next_id: parent_condition.true_next_id,
                    false_next_id: parent_condition.false_next_id,
                };
                (lhs, rhs)
            }
        };
        // We visit expressions tree in pre-order, so place the left-hand side on the top.
        decision_ctx.decision_stack.push_back(rhs);
        decision_ctx.decision_stack.push_back(lhs);
    }

    fn try_finish_decision(
        &mut self,
        span: Span,
        true_marker: BlockMarkerId,
        false_marker: BlockMarkerId,
        degraded_branches: &mut Vec<MCDCBranchSpan>,
    ) -> Option<(MCDCDecisionSpan, Vec<MCDCBranchSpan>)> {
        let Some(decision_ctx) = self.decision_ctx_stack.last_mut() else {
            bug!("Unexpected empty decision_ctx_stack")
        };
        let Some(condition_info) = decision_ctx.decision_stack.pop_back() else {
            let branch = MCDCBranchSpan {
                span,
                condition_info: ConditionInfo {
                    condition_id: ConditionId::START,
                    true_next_id: None,
                    false_next_id: None,
                },
                true_marker,
                false_marker,
            };
            degraded_branches.push(branch);
            return None;
        };
        let Some(decision) = decision_ctx.processing_decision.as_mut() else {
            bug!("Processing decision should have been created before any conditions are taken");
        };
        if condition_info.true_next_id.is_none() {
            decision.end_markers.push(true_marker);
        }
        if condition_info.false_next_id.is_none() {
            decision.end_markers.push(false_marker);
        }
        decision_ctx.conditions.push(MCDCBranchSpan {
            span,
            condition_info,
            true_marker,
            false_marker,
        });

        if decision_ctx.decision_stack.is_empty() {
            let conditions = std::mem::take(&mut decision_ctx.conditions);
            decision_ctx.processing_decision.take().map(|decision| (decision, conditions))
        } else {
            None
        }
    }
}

pub(crate) struct MCDCInfoBuilder {
    degraded_spans: Vec<MCDCBranchSpan>,
    mcdc_spans: Vec<(MCDCDecisionSpan, Vec<MCDCBranchSpan>)>,
    state: MCDCState,
}

impl MCDCInfoBuilder {
    pub(crate) fn new() -> Self {
        Self { degraded_spans: vec![], mcdc_spans: vec![], state: MCDCState::new() }
    }

    pub(crate) fn visit_evaluated_condition(
        &mut self,
        tcx: TyCtxt<'_>,
        source_info: SourceInfo,
        true_block: BasicBlock,
        false_block: BasicBlock,
        mut inject_block_marker: impl FnMut(SourceInfo, BasicBlock) -> BlockMarkerId,
    ) {
        let true_marker = inject_block_marker(source_info, true_block);
        let false_marker = inject_block_marker(source_info, false_block);

        // take_condition() returns Some for decision_result when the decision stack
        // is empty, i.e. when all the conditions of the decision were instrumented,
        // and the decision is "complete".
        if let Some((decision, conditions)) = self.state.try_finish_decision(
            source_info.span,
            true_marker,
            false_marker,
            &mut self.degraded_spans,
        ) {
            let num_conditions = conditions.len();
            assert_eq!(
                num_conditions, decision.num_conditions,
                "final number of conditions is not correct"
            );
            match num_conditions {
                0 => {
                    unreachable!("Decision with no condition is not expected");
                }
                1..=MAX_CONDITIONS_IN_DECISION => {
                    self.mcdc_spans.push((decision, conditions));
                }
                _ => {
                    self.degraded_spans.extend(conditions);

                    tcx.dcx().emit_warn(MCDCExceedsConditionLimit {
                        span: decision.span,
                        num_conditions,
                        max_conditions: MAX_CONDITIONS_IN_DECISION,
                    });
                }
            }
        }
    }

    pub(crate) fn into_done(
        self,
    ) -> (Vec<(MCDCDecisionSpan, Vec<MCDCBranchSpan>)>, Vec<MCDCBranchSpan>) {
        (self.mcdc_spans, self.degraded_spans)
    }
}

impl Builder<'_, '_> {
    pub(crate) fn visit_coverage_branch_operation(&mut self, logical_op: LogicalOp, span: Span) {
        if let Some(coverage_info) = self.coverage_info.as_mut()
            && let Some(mcdc_info) = coverage_info.mcdc_info.as_mut()
        {
            mcdc_info.state.record_conditions(logical_op, span);
        }
    }

    pub(crate) fn mcdc_increment_depth_if_enabled(&mut self) {
        if let Some(coverage_info) = self.coverage_info.as_mut()
            && let Some(mcdc_info) = coverage_info.mcdc_info.as_mut()
        {
            mcdc_info.state.decision_ctx_stack.push(MCDCDecisionCtx::default());
        };
    }

    pub(crate) fn mcdc_decrement_depth_if_enabled(&mut self) {
        if let Some(coverage_info) = self.coverage_info.as_mut()
            && let Some(mcdc_info) = coverage_info.mcdc_info.as_mut()
            && mcdc_info.state.decision_ctx_stack.pop().is_none()
        {
            bug!("Unexpected empty decision stack");
        };
    }
}
