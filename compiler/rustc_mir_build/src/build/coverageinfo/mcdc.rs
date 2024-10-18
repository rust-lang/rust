mod matching;
use std::cell::Cell;
use std::collections::VecDeque;
use std::rc::Rc;

use matching::{LateMatchingState, MatchingDecisionCtx};
use rustc_data_structures::fx::FxIndexMap;
use rustc_middle::bug;
use rustc_middle::mir::BasicBlock;
use rustc_middle::mir::coverage::{
    BlockMarkerId, ConditionId, ConditionInfo, DecisionId, MCDCBranchSpan, MCDCDecisionSpan,
};
use rustc_middle::thir::{ExprKind, LogicalOp};
use rustc_middle::ty::TyCtxt;
use rustc_span::Span;

use crate::build::Builder;
use crate::errors::{MCDCExceedsConditionLimit, MCDCExceedsDecisionDepth};

/// LLVM uses `i16` to represent condition id. Hence `i16::MAX` is the hard limit for number of
/// conditions in a decision.
const MAX_CONDITIONS_IN_DECISION: usize = i16::MAX as usize;

/// MCDC allocates an i32 variable on stack for each depth. Ignore decisions nested too much to prevent it
/// consuming excessive memory.
const MAX_DECISION_DEPTH: u16 = 0x3FFF;

#[derive(Debug)]
struct BooleanDecisionCtx {
    id: DecisionId,
    decision_info: MCDCDecisionSpan,
    /// To construct condition evaluation tree.
    decision_stack: VecDeque<ConditionInfo>,
    conditions: Vec<MCDCBranchSpan>,
    condition_id_counter: usize,
}

impl BooleanDecisionCtx {
    fn new(id: DecisionId) -> Self {
        Self {
            id,
            decision_info: MCDCDecisionSpan::new(Span::default()),
            decision_stack: VecDeque::new(),
            conditions: vec![],
            condition_id_counter: 0,
        }
    }

    fn next_condition_id(&mut self) -> ConditionId {
        let id = ConditionId::from_usize(self.condition_id_counter);
        self.condition_id_counter += 1;
        id
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
    fn record_conditions(&mut self, op: LogicalOp) {
        let parent_condition = match self.decision_stack.pop_back() {
            Some(info) => info,
            None => ConditionInfo {
                condition_id: self.next_condition_id(),
                true_next_id: None,
                false_next_id: None,
            },
        };
        let lhs_id = parent_condition.condition_id;

        let rhs_condition_id = self.next_condition_id();

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
        self.decision_stack.push_back(rhs);
        self.decision_stack.push_back(lhs);
    }

    fn finish_two_way_branch(
        &mut self,
        span: Span,
        true_marker: BlockMarkerId,
        false_marker: BlockMarkerId,
    ) {
        let condition_info = self.decision_stack.pop_back().unwrap_or(ConditionInfo {
            condition_id: ConditionId::START,
            true_next_id: None,
            false_next_id: None,
        });
        if condition_info.true_next_id.is_none() {
            self.decision_info.update_end_markers.push(true_marker);
        }
        if condition_info.false_next_id.is_none() {
            self.decision_info.update_end_markers.push(false_marker);
        }

        self.conditions.push(MCDCBranchSpan {
            span,
            condition_info,
            true_markers: vec![true_marker],
            false_markers: vec![false_marker],
        });
    }

    fn is_finished(&self) -> bool {
        self.decision_stack.is_empty()
    }

    fn into_done(self) -> (DecisionId, MCDCDecisionSpan, Vec<MCDCBranchSpan>) {
        (self.id, self.decision_info, self.conditions)
    }
}

#[derive(Debug)]
enum DecisionCtx {
    Boolean(BooleanDecisionCtx),
    Matching(MatchingDecisionCtx),
}

impl DecisionCtx {
    fn new_boolean(id: DecisionId) -> Self {
        Self::Boolean(BooleanDecisionCtx::new(id))
    }

    fn new_matching(info: &[(Span, DecisionId)]) -> Self {
        Self::Matching(MatchingDecisionCtx::new(info))
    }
}

pub(crate) struct MCDCStateGuard {
    state_stashed_ref: Option<Rc<Cell<bool>>>,
}

impl Drop for MCDCStateGuard {
    fn drop(&mut self) {
        if let Some(stashed) = self.state_stashed_ref.take() {
            stashed.set(false);
        }
    }
}

/// `MCDCState` represents a layer to hold decisions. Decisions produced
/// by same state are nested in same decision.
#[derive(Debug)]
struct MCDCState {
    current_ctx: Option<DecisionCtx>,
    nested_decision_records: Vec<DecisionId>,
    // `Stashed` means we are processing a decision nested in decision of this state.
    stashed: Rc<Cell<bool>>,
}

impl MCDCState {
    fn new() -> Self {
        Self {
            current_ctx: None,
            nested_decision_records: vec![],
            stashed: Rc::new(Cell::new(false)),
        }
    }

    fn is_stashed(&self) -> bool {
        self.stashed.get()
    }

    fn take_ctx(&mut self) -> Option<(DecisionCtx, Vec<DecisionId>)> {
        let ctx = self.current_ctx.take()?;
        let nested_decisions_id = std::mem::take(&mut self.nested_decision_records);
        Some((ctx, nested_decisions_id))
    }

    // Return `true` if there is no decision being processed currently.
    fn is_empty(&self) -> bool {
        self.current_ctx.is_none()
    }

    fn record_nested_decision(&mut self, id: DecisionId) {
        self.nested_decision_records.push(id);
    }

    fn inherit_nested_decisions(&mut self, nested_decisions_id: Vec<DecisionId>) {
        self.nested_decision_records.extend(nested_decisions_id);
    }

    fn take_current_nested_decisions(&mut self) -> Vec<DecisionId> {
        std::mem::take(&mut self.nested_decision_records)
    }
}

#[derive(Debug)]
struct MCDCTargetInfo {
    decision: MCDCDecisionSpan,
    conditions: Vec<MCDCBranchSpan>,
    nested_decisions_id: Vec<DecisionId>,
}

impl MCDCTargetInfo {
    fn set_depth(&mut self, depth: u16) {
        self.decision.decision_depth = depth;
    }
}

#[derive(Default)]
struct DecisionIdGen(usize);
impl DecisionIdGen {
    fn next_decision_id(&mut self) -> DecisionId {
        let id = DecisionId::from_usize(self.0);
        self.0 += 1;
        id
    }
}

pub(crate) struct MCDCInfoBuilder {
    normal_branch_spans: Vec<MCDCBranchSpan>,
    mcdc_targets: FxIndexMap<DecisionId, MCDCTargetInfo>,
    state_stack: Vec<MCDCState>,
    late_matching_state: LateMatchingState,
    decision_id_gen: DecisionIdGen,
}

impl MCDCInfoBuilder {
    pub(crate) fn new() -> Self {
        Self {
            normal_branch_spans: vec![],
            mcdc_targets: FxIndexMap::default(),
            state_stack: vec![],
            late_matching_state: Default::default(),
            decision_id_gen: DecisionIdGen::default(),
        }
    }

    fn has_processing_decision(&self) -> bool {
        // Check from top to get working states a bit quicker.
        !self.state_stack.iter().rev().all(|state| state.is_empty() && !state.is_stashed())
    }

    fn current_state_mut(&mut self) -> &mut MCDCState {
        let current_idx = self.state_stack.len() - 1;
        &mut self.state_stack[current_idx]
    }

    fn current_processing_ctx_mut(&mut self) -> Option<&mut DecisionCtx> {
        self.ensure_active_state();
        self.state_stack.last_mut().and_then(|state| state.current_ctx.as_mut())
    }

    fn ensure_active_state(&mut self) {
        let mut active_state_idx = None;
        // Down to the first non-stashed state or non-empty state, which can be ensured to be
        // processed currently.
        for (idx, state) in self.state_stack.iter().enumerate().rev() {
            if state.is_stashed() {
                active_state_idx = Some(idx + 1);
                break;
            } else if !state.is_empty() {
                active_state_idx = Some(idx);
                break;
            }
        }
        match active_state_idx {
            // There are some states were created for nested decisions but now
            // since the lower state has been unstashed they should be removed.
            Some(idx) if idx + 1 < self.state_stack.len() => {
                let expected_len = idx + 1;
                let nested_decisions_id = self
                    .state_stack
                    .iter_mut()
                    .skip(expected_len)
                    .map(|state| state.take_current_nested_decisions().into_iter())
                    .flatten()
                    .collect();
                self.state_stack.truncate(expected_len);
                self.state_stack[idx].inherit_nested_decisions(nested_decisions_id);
            }
            // The top state is just wanted.
            Some(idx) if idx + 1 == self.state_stack.len() => {}
            // Otherwise no available state yet, create a new one.
            _ => self.state_stack.push(MCDCState::new()),
        }
    }

    fn ensure_boolean_decision(&mut self, condition_span: Span) -> &mut BooleanDecisionCtx {
        self.ensure_active_state();
        let state = self.state_stack.last_mut().expect("ensured just now");
        let DecisionCtx::Boolean(ctx) = state.current_ctx.get_or_insert_with(|| {
            DecisionCtx::new_boolean(self.decision_id_gen.next_decision_id())
        }) else {
            unreachable!("ensured above");
        };

        if ctx.decision_info.span == Span::default() {
            ctx.decision_info.span = condition_span;
        } else {
            ctx.decision_info.span = ctx.decision_info.span.to(condition_span);
        }
        ctx
    }

    fn append_normal_branches(&mut self, branches: Vec<MCDCBranchSpan>) {
        self.normal_branch_spans.extend(branches);
    }

    fn append_mcdc_info(&mut self, tcx: TyCtxt<'_>, id: DecisionId, info: MCDCTargetInfo) -> bool {
        let num_conditions = info.conditions.len();
        match num_conditions {
            0 => {
                // Irrefutable patterns caused by empty types can lead to here.
                false
            }
            // Ignore decisions with only one condition given that mcdc for them is completely equivalent to branch coverage.
            2..=MAX_CONDITIONS_IN_DECISION => {
                self.mcdc_targets.insert(id, info);
                true
            }
            _ => {
                self.append_normal_branches(info.conditions);
                self.current_state_mut().inherit_nested_decisions(info.nested_decisions_id);
                if num_conditions > MAX_CONDITIONS_IN_DECISION {
                    tcx.dcx().emit_warn(MCDCExceedsConditionLimit {
                        span: info.decision.span,
                        num_conditions,
                        max_conditions: MAX_CONDITIONS_IN_DECISION,
                    });
                }
                false
            }
        }
    }

    fn normalize_depth_from(&mut self, tcx: TyCtxt<'_>, id: DecisionId) {
        let Some(entry_decision) = self.mcdc_targets.get_mut(&id) else {
            bug!("unknown mcdc decision");
        };
        let mut next_nested_records = entry_decision.nested_decisions_id.clone();
        let mut depth = 0;
        while !next_nested_records.is_empty() {
            depth += 1;
            for id in std::mem::take(&mut next_nested_records) {
                let Some(nested_target) = self.mcdc_targets.get_mut(&id) else {
                    continue;
                };
                nested_target.set_depth(depth);
                next_nested_records.extend(nested_target.nested_decisions_id.iter().copied());
                if depth > MAX_DECISION_DEPTH {
                    tcx.dcx().emit_warn(MCDCExceedsDecisionDepth {
                        span: nested_target.decision.span,
                        max_decision_depth: MAX_DECISION_DEPTH.into(),
                    });
                    let branches = std::mem::take(&mut nested_target.conditions);
                    self.append_normal_branches(branches);
                    self.mcdc_targets.swap_remove(&id);
                }
            }
        }
    }

    // If `entry_decision_id` is some, there must be at least one mcdc decision being produced.
    fn on_ctx_finished(&mut self, tcx: TyCtxt<'_>, entry_decision_id: Option<DecisionId>) {
        match (self.has_processing_decision(), entry_decision_id) {
            // Can not be nested in other decisions, depth is accumulated starting from this decision.
            (false, Some(id)) => self.normalize_depth_from(tcx, id),
            // May be nested in other decisions, record it.
            (true, Some(id)) => self.current_state_mut().record_nested_decision(id),
            // No decision is produced this time and no other parent decision to be processing.
            // All "nested decisions" now get zero depth and then calculate depth of their children.
            (false, None) => {
                for root_decision in self.current_state_mut().take_current_nested_decisions() {
                    self.normalize_depth_from(tcx, root_decision);
                }
            }
            (true, None) => {}
        }
    }

    pub(crate) fn visit_evaluated_condition(
        &mut self,
        tcx: TyCtxt<'_>,
        span: Span,
        true_block: BasicBlock,
        false_block: BasicBlock,
        mut inject_block_marker: impl FnMut(BasicBlock) -> BlockMarkerId,
    ) {
        let true_marker = inject_block_marker(true_block);
        let false_marker = inject_block_marker(false_block);
        let decision = self.ensure_boolean_decision(span);
        decision.finish_two_way_branch(span, true_marker, false_marker);

        if !decision.is_finished() {
            return;
        }

        let Some((DecisionCtx::Boolean(ctx), nested_decisions_id)) =
            self.current_state_mut().take_ctx()
        else {
            unreachable!("ensured boolean ctx above");
        };

        let (id, decision, conditions) = ctx.into_done();
        let info = MCDCTargetInfo { decision, conditions, nested_decisions_id };
        if self.late_matching_state.is_guard_decision(id) {
            self.late_matching_state.add_guard_decision(id, info);
        } else {
            let entry_id = self.append_mcdc_info(tcx, id, info).then_some(id);
            self.on_ctx_finished(tcx, entry_id)
        }
    }

    pub(crate) fn into_done(
        self,
    ) -> (Vec<MCDCBranchSpan>, Vec<(MCDCDecisionSpan, Vec<MCDCBranchSpan>)>) {
        assert!(
            !self.has_processing_decision() && self.late_matching_state.is_empty(),
            "has unfinished decisions"
        );
        let MCDCInfoBuilder {
            normal_branch_spans,
            mcdc_targets,
            state_stack: _,
            late_matching_state: _,
            decision_id_gen: _,
        } = self;

        let mcdc_spans = mcdc_targets
            .into_values()
            .map(|MCDCTargetInfo { decision, conditions, nested_decisions_id: _ }| {
                (decision, conditions)
            })
            .collect();

        (normal_branch_spans, mcdc_spans)
    }
}

impl Builder<'_, '_> {
    pub(crate) fn visit_coverage_branch_operation(&mut self, logical_op: LogicalOp, span: Span) {
        if let Some(coverage_info) = self.coverage_info.as_mut()
            && let Some(mcdc_info) = coverage_info.mcdc_info.as_mut()
        {
            let decision = mcdc_info.ensure_boolean_decision(span);
            decision.record_conditions(logical_op);
        }
    }

    pub(crate) fn mcdc_prepare_ctx_for(&mut self, expr_kind: &ExprKind<'_>) -> MCDCStateGuard {
        if let Some(coverage_info) = self.coverage_info.as_mut()
            && let Some(mcdc_info) = coverage_info.mcdc_info.as_mut()
        {
            match expr_kind {
                ExprKind::Unary { .. } | ExprKind::Scope { .. } => {}
                ExprKind::LogicalOp { .. } => {
                    // By here a decision is going to be produced
                    mcdc_info.ensure_active_state();
                }
                _ => {
                    // Non-logical expressions leads to nested decisions only when a decision is being processed.
                    // In such cases just mark the state `stashed`. If a nested decision following, a new active state will be
                    // created at the previous arm. The current top state will be unstashed when the guard is dropped.
                    if mcdc_info.has_processing_decision() {
                        let stashed = &mcdc_info.current_state_mut().stashed;
                        if !stashed.replace(true) {
                            return MCDCStateGuard { state_stashed_ref: Some(stashed.clone()) };
                        }
                    }
                }
            }
        };
        MCDCStateGuard { state_stashed_ref: None }
    }
}
