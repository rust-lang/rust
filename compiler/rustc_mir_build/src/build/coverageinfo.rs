use std::assert_matches::assert_matches;
use std::collections::hash_map::Entry;
use std::collections::VecDeque;

use rustc_data_structures::fx::FxHashMap;
use rustc_middle::mir::coverage::{
    BlockMarkerId, BranchSpan, ConditionId, ConditionInfo, CoverageKind, MCDCBranchSpan,
    MCDCDecisionSpan,
};
use rustc_middle::mir::{self, BasicBlock, SourceInfo, UnOp};
use rustc_middle::thir::{ExprId, ExprKind, LogicalOp, Thir};
use rustc_middle::ty::TyCtxt;
use rustc_span::def_id::LocalDefId;
use rustc_span::Span;

use crate::build::{Builder, CFG};
use crate::errors::MCDCExceedsConditionNumLimit;

pub(crate) struct BranchInfoBuilder {
    /// Maps condition expressions to their enclosing `!`, for better instrumentation.
    nots: FxHashMap<ExprId, NotInfo>,

    num_block_markers: usize,
    branch_spans: Vec<BranchSpan>,

    mcdc_branch_spans: Vec<MCDCBranchSpan>,
    mcdc_decision_spans: Vec<MCDCDecisionSpan>,
    mcdc_state: Option<MCDCState>,
}

#[derive(Clone, Copy)]
struct NotInfo {
    /// When visiting the associated expression as a branch condition, treat this
    /// enclosing `!` as the branch condition instead.
    enclosing_not: ExprId,
    /// True if the associated expression is nested within an odd number of `!`
    /// expressions relative to `enclosing_not` (inclusive of `enclosing_not`).
    is_flipped: bool,
}

impl BranchInfoBuilder {
    /// Creates a new branch info builder, but only if branch coverage instrumentation
    /// is enabled and `def_id` represents a function that is eligible for coverage.
    pub(crate) fn new_if_enabled(tcx: TyCtxt<'_>, def_id: LocalDefId) -> Option<Self> {
        if tcx.sess.instrument_coverage_branch() && tcx.is_eligible_for_coverage(def_id) {
            Some(Self {
                nots: FxHashMap::default(),
                num_block_markers: 0,
                branch_spans: vec![],
                mcdc_branch_spans: vec![],
                mcdc_decision_spans: vec![],
                mcdc_state: MCDCState::new_if_enabled(tcx),
            })
        } else {
            None
        }
    }

    /// Unary `!` expressions inside an `if` condition are lowered by lowering
    /// their argument instead, and then reversing the then/else arms of that `if`.
    ///
    /// That's awkward for branch coverage instrumentation, so to work around that
    /// we pre-emptively visit any affected `!` expressions, and record extra
    /// information that [`Builder::visit_coverage_branch_condition`] can use to
    /// synthesize branch instrumentation for the enclosing `!`.
    pub(crate) fn visit_unary_not(&mut self, thir: &Thir<'_>, unary_not: ExprId) {
        assert_matches!(thir[unary_not].kind, ExprKind::Unary { op: UnOp::Not, .. });

        self.visit_with_not_info(
            thir,
            unary_not,
            // Set `is_flipped: false` for the `!` itself, so that its enclosed
            // expression will have `is_flipped: true`.
            NotInfo { enclosing_not: unary_not, is_flipped: false },
        );
    }

    fn visit_with_not_info(&mut self, thir: &Thir<'_>, expr_id: ExprId, not_info: NotInfo) {
        match self.nots.entry(expr_id) {
            // This expression has already been marked by an enclosing `!`.
            Entry::Occupied(_) => return,
            Entry::Vacant(entry) => entry.insert(not_info),
        };

        match thir[expr_id].kind {
            ExprKind::Unary { op: UnOp::Not, arg } => {
                // Invert the `is_flipped` flag for the contents of this `!`.
                let not_info = NotInfo { is_flipped: !not_info.is_flipped, ..not_info };
                self.visit_with_not_info(thir, arg, not_info);
            }
            ExprKind::Scope { value, .. } => self.visit_with_not_info(thir, value, not_info),
            ExprKind::Use { source } => self.visit_with_not_info(thir, source, not_info),
            // All other expressions (including `&&` and `||`) don't need any
            // special handling of their contents, so stop visiting.
            _ => {}
        }
    }

    fn fetch_mcdc_condition_info(
        &mut self,
        tcx: TyCtxt<'_>,
        true_marker: BlockMarkerId,
        false_marker: BlockMarkerId,
    ) -> Option<ConditionInfo> {
        let mcdc_state = self.mcdc_state.as_mut()?;
        let (mut condition_info, decision_result) =
            mcdc_state.take_condition(true_marker, false_marker);
        if let Some(decision) = decision_result {
            match decision.conditions_num {
                0 => {
                    unreachable!("Decision with no condition is not expected");
                }
                1..=MAX_CONDITIONS_NUM_IN_DECISION => {
                    self.mcdc_decision_spans.push(decision);
                }
                _ => {
                    // Do not generate mcdc mappings and statements for decisions with too many conditions.
                    let rebase_idx = self.mcdc_branch_spans.len() - decision.conditions_num + 1;
                    for branch in &mut self.mcdc_branch_spans[rebase_idx..] {
                        branch.condition_info = None;
                    }

                    // ConditionInfo of this branch shall also be reset.
                    condition_info = None;

                    tcx.dcx().emit_warn(MCDCExceedsConditionNumLimit {
                        span: decision.span,
                        conditions_num: decision.conditions_num,
                        max_conditions_num: MAX_CONDITIONS_NUM_IN_DECISION,
                    });
                }
            }
        }
        condition_info
    }

    fn add_two_way_branch<'tcx>(
        &mut self,
        cfg: &mut CFG<'tcx>,
        source_info: SourceInfo,
        true_block: BasicBlock,
        false_block: BasicBlock,
    ) {
        let true_marker = self.inject_block_marker(cfg, source_info, true_block);
        let false_marker = self.inject_block_marker(cfg, source_info, false_block);

        self.branch_spans.push(BranchSpan { span: source_info.span, true_marker, false_marker });
    }

    fn next_block_marker_id(&mut self) -> BlockMarkerId {
        let id = BlockMarkerId::from_usize(self.num_block_markers);
        self.num_block_markers += 1;
        id
    }

    fn inject_block_marker(
        &mut self,
        cfg: &mut CFG<'_>,
        source_info: SourceInfo,
        block: BasicBlock,
    ) -> BlockMarkerId {
        let id = self.next_block_marker_id();

        let marker_statement = mir::Statement {
            source_info,
            kind: mir::StatementKind::Coverage(CoverageKind::BlockMarker { id }),
        };
        cfg.push(block, marker_statement);

        id
    }

    pub(crate) fn into_done(self) -> Option<Box<mir::coverage::BranchInfo>> {
        let Self {
            nots: _,
            num_block_markers,
            branch_spans,
            mcdc_branch_spans,
            mcdc_decision_spans,
            mcdc_state: _,
        } = self;

        if num_block_markers == 0 {
            assert!(branch_spans.is_empty());
            return None;
        }

        Some(Box::new(mir::coverage::BranchInfo {
            num_block_markers,
            branch_spans,
            mcdc_branch_spans,
            mcdc_decision_spans,
        }))
    }
}

/// The MCDC bitmap scales exponentially (2^n) based on the number of conditions seen,
/// So llvm sets a maximum value prevents the bitmap footprint from growing too large without the user's knowledge.
/// This limit may be relaxed if the [upstream change](https://github.com/llvm/llvm-project/pull/82448) is merged.
const MAX_CONDITIONS_NUM_IN_DECISION: usize = 6;

struct MCDCState {
    /// To construct condition evaluation tree.
    decision_stack: VecDeque<ConditionInfo>,
    processing_decision: Option<MCDCDecisionSpan>,
}

impl MCDCState {
    fn new_if_enabled(tcx: TyCtxt<'_>) -> Option<Self> {
        tcx.sess
            .instrument_coverage_mcdc()
            .then(|| Self { decision_stack: VecDeque::new(), processing_decision: None })
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
        let decision = match self.processing_decision.as_mut() {
            Some(decision) => {
                decision.span = decision.span.to(span);
                decision
            }
            None => self.processing_decision.insert(MCDCDecisionSpan {
                span,
                conditions_num: 0,
                end_markers: vec![],
            }),
        };

        let parent_condition = self.decision_stack.pop_back().unwrap_or_default();
        let lhs_id = if parent_condition.condition_id == ConditionId::NONE {
            decision.conditions_num += 1;
            ConditionId::from(decision.conditions_num)
        } else {
            parent_condition.condition_id
        };

        decision.conditions_num += 1;
        let rhs_condition_id = ConditionId::from(decision.conditions_num);

        let (lhs, rhs) = match op {
            LogicalOp::And => {
                let lhs = ConditionInfo {
                    condition_id: lhs_id,
                    true_next_id: rhs_condition_id,
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
                    false_next_id: rhs_condition_id,
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
        self.decision_stack.push_back(rhs);
        self.decision_stack.push_back(lhs);
    }

    fn take_condition(
        &mut self,
        true_marker: BlockMarkerId,
        false_marker: BlockMarkerId,
    ) -> (Option<ConditionInfo>, Option<MCDCDecisionSpan>) {
        let Some(condition_info) = self.decision_stack.pop_back() else {
            return (None, None);
        };
        let Some(decision) = self.processing_decision.as_mut() else {
            bug!("Processing decision should have been created before any conditions are taken");
        };
        if condition_info.true_next_id == ConditionId::NONE {
            decision.end_markers.push(true_marker);
        }
        if condition_info.false_next_id == ConditionId::NONE {
            decision.end_markers.push(false_marker);
        }

        if self.decision_stack.is_empty() {
            (Some(condition_info), self.processing_decision.take())
        } else {
            (Some(condition_info), None)
        }
    }
}

impl Builder<'_, '_> {
    /// If branch coverage is enabled, inject marker statements into `then_block`
    /// and `else_block`, and record their IDs in the table of branch spans.
    pub(crate) fn visit_coverage_branch_condition(
        &mut self,
        mut expr_id: ExprId,
        mut then_block: BasicBlock,
        mut else_block: BasicBlock,
    ) {
        // Bail out if branch coverage is not enabled for this function.
        let Some(branch_info) = self.coverage_branch_info.as_mut() else { return };

        // If this condition expression is nested within one or more `!` expressions,
        // replace it with the enclosing `!` collected by `visit_unary_not`.
        if let Some(&NotInfo { enclosing_not, is_flipped }) = branch_info.nots.get(&expr_id) {
            expr_id = enclosing_not;
            if is_flipped {
                std::mem::swap(&mut then_block, &mut else_block);
            }
        }

        let source_info = SourceInfo { span: self.thir[expr_id].span, scope: self.source_scope };

        // Separate path for handling branches when MC/DC is enabled.
        if branch_info.mcdc_state.is_some() {
            let mut inject_block_marker =
                |block| branch_info.inject_block_marker(&mut self.cfg, source_info, block);
            let true_marker = inject_block_marker(then_block);
            let false_marker = inject_block_marker(else_block);
            let condition_info =
                branch_info.fetch_mcdc_condition_info(self.tcx, true_marker, false_marker);
            branch_info.mcdc_branch_spans.push(MCDCBranchSpan {
                span: source_info.span,
                condition_info,
                true_marker,
                false_marker,
            });
            return;
        }

        branch_info.add_two_way_branch(&mut self.cfg, source_info, then_block, else_block);
    }

    pub(crate) fn visit_coverage_branch_operation(&mut self, logical_op: LogicalOp, span: Span) {
        if let Some(branch_info) = self.coverage_branch_info.as_mut()
            && let Some(mcdc_state) = branch_info.mcdc_state.as_mut()
        {
            mcdc_state.record_conditions(logical_op, span);
        }
    }
}
