use rustc_middle::bug;
use rustc_middle::mir::coverage::{
    BlockMarkerId, ConditionId, ConditionInfo, MCDCConditionSpan, MCDCDecisionSpan,
};
use rustc_middle::thir::LogicalOp;
use rustc_middle::ty::TyCtxt;
use rustc_span::Span;
use tracing::debug;

use crate::builder::Builder;
use crate::errors::MCDCNumberConditionExceeded;

/// This struct is responsible for assigning [`ConditionId`]s to conditions and
/// building the BDD (Binary Decision Diagram) of a decision, which are
/// necessary for generating the MCDC coverage mappings.
///
/// The ID assignment algorithm works by assigning an ID to the operands of the
/// root operator, and then, via consecutive calls on the sub-expressions, if
/// its a composite expression, re-assign its ID to its LHS, and assign a new
/// ID to its RHS.
///
/// Example: "x = (A && B) || (C && D) || (D && F)"
///
///      Visit Depth1:
///              (A && B) || (C && D) || (D && F)
///              ^-------LHS--------^    ^-RHS--^
///                      ID=1              ID=2
///
///      Visit LHS-Depth2:
///              (A && B) || (C && D)
///              ^-LHS--^    ^-RHS--^
///                ID=1        ID=3
///
///      Visit LHS-Depth3:
///               (A && B)
///               LHS   RHS
///               ID=1  ID=4
///
///      Visit RHS-Depth3:
///                         (C && D)
///                         LHS   RHS
///                         ID=3  ID=5
///
///      Visit RHS-Depth2:              (D && F)
///                                     LHS   RHS
///                                     ID=2  ID=6
///
///      Visit Depth1:
///              (A && B)  || (C && D)  || (D && F)
///              ID=1  ID=4   ID=3  ID=5   ID=2  ID=6
///
#[derive(Debug, Default)]
struct MCDCDecisionBuilder {
    bdd_node_stack: Vec<ConditionInfo>,
    conditions: Vec<MCDCConditionSpan>,
    decision: Option<MCDCDecisionSpan>,
}

impl MCDCDecisionBuilder {
    /// This function assigns a [`ConditionId`] to each condition of a decision
    /// through consecutive calls on each logical operator within a boolean
    /// expression, and builds the BDD of the current decision.
    fn record_operator(&mut self, logical_op: LogicalOp, span: Span, decision_depth: u16) {
        debug!("record {logical_op:?} operator for MCDC at {span:?}");

        let decision = match self.decision.as_mut() {
            Some(decision) => {
                decision.span = decision.span.to(span);
                decision
            }
            None => self.decision.insert(MCDCDecisionSpan {
                span,
                end_markers: Vec::new(),
                decision_depth,
                num_conditions: 0,
            }),
        };

        let parent_condition = self.bdd_node_stack.pop().unwrap_or_else(|| {
            assert_eq!(decision.num_conditions, 0, "No condition means num_conditions should be 0");
            decision.num_conditions += 1;
            ConditionInfo {
                condition_id: ConditionId::START,
                true_next_id: None,
                false_next_id: None,
            }
        });

        let lhs_id = parent_condition.condition_id;
        let rhs_id = ConditionId::from(decision.num_conditions);
        decision.num_conditions += 1;

        let lhs = match logical_op {
            LogicalOp::And => ConditionInfo {
                condition_id: lhs_id,
                true_next_id: Some(rhs_id),
                false_next_id: parent_condition.false_next_id,
            },
            LogicalOp::Or => ConditionInfo {
                condition_id: lhs_id,
                true_next_id: parent_condition.true_next_id,
                false_next_id: Some(rhs_id),
            },
        };
        let rhs = ConditionInfo {
            condition_id: rhs_id,
            true_next_id: parent_condition.true_next_id,
            false_next_id: parent_condition.false_next_id,
        };

        self.bdd_node_stack.push(rhs);
        self.bdd_node_stack.push(lhs);
    }

    /// Pop the [`ConditionInfo`] at the top of the node stack and make an
    /// [`MCDCConditionSpan`] from it. If the stack is empty afterwards, it
    /// means the decision was completely visited, so return the decision and
    /// condition spans.
    fn try_finish_decision(
        &mut self,
        span: Span,
        true_marker: BlockMarkerId,
        false_marker: BlockMarkerId,
    ) -> Option<(MCDCDecisionSpan, Vec<MCDCConditionSpan>)> {
        let condition_info = self.bdd_node_stack.pop()?;
        let Some(decision) = self.decision.as_mut() else {
            bug!("decision should have been created by now");
        };

        // if true_next_id is None, it means that this condition's evaluation
        // to true will make the decision true.
        if condition_info.true_next_id.is_none() {
            decision.end_markers.push(true_marker);
        }
        if condition_info.false_next_id.is_none() {
            decision.end_markers.push(false_marker);
        }

        // The condition_info is a leaf condition, make it a branch span.
        self.conditions.push(MCDCConditionSpan { span, condition_info, true_marker, false_marker });

        if self.bdd_node_stack.is_empty() {
            // There is no more condition to the decision, return the resulting spans
            let conditions = std::mem::take(&mut self.conditions);
            self.decision.take().map(|decision| (decision, conditions))
        } else {
            None
        }
    }

    /// Returns true if the builder is currently building a decision.
    #[inline]
    fn is_empty(&self) -> bool {
        return self.decision.is_none();
    }
}

/// MCDC decision builder for a function.
#[derive(Debug)]
pub(crate) struct MCDCInfoBuilder {
    /// Hold a stack of decision builder, to account for nested decisions.
    ///
    /// Example:
    /// ```rust,ignore (illustrative)
    /// if a && foo(b || c) {}
    /// ```
    ///
    /// Here `b || c` is a different decision nested in `a && ...`.
    /// To avoid mixing decisions, when we're building a decision and we visit
    /// a "non-trivial" node (other than parentheses, deref, etc...), we
    /// push a new builder on the stack.
    decision_builder_stack: Vec<MCDCDecisionBuilder>,
}

impl Default for MCDCInfoBuilder {
    fn default() -> Self {
        Self { decision_builder_stack: vec![MCDCDecisionBuilder::default()] }
    }
}

impl MCDCInfoBuilder {
    /// Decision depth is given as a u16 to reduce the size of the
    /// `CoverageKind`, as it is very unlikely that the depth ever reaches
    /// 2^16.
    #[inline]
    fn decision_depth(&self) -> u16 {
        match u16::try_from(self.decision_builder_stack.len())
            .expect(
                "decision depth did not fit in u16, this is likely to be an instrumentation error",
            )
            .checked_sub(1)
        {
            Some(d) => d,
            None => bug!("Unexpected empty decision builder stack"),
        }
    }

    pub(crate) fn record_operator(&mut self, logical_op: LogicalOp, span: Span) {
        let depth = self.decision_depth();

        let Some(decision_builder) = self.decision_builder_stack.last_mut() else {
            bug!("Unexpected empty decision builder stack")
        };

        decision_builder.record_operator(logical_op, span, depth);
    }

    pub(crate) fn record_leaf_condition(
        &mut self,
        tcx: TyCtxt<'_>,
        span: Span,
        true_marker: BlockMarkerId,
        false_marker: BlockMarkerId,
    ) {
        let Some(decision_builder) = self.decision_builder_stack.last_mut() else {
            bug!("Unexpected empty decision builder stack")
        };

        if let Some((decision, conditions)) =
            decision_builder.try_finish_decision(span, true_marker, false_marker)
        {
            let num_conditions = conditions.len();
            assert_eq!(
                num_conditions, decision.num_conditions,
                "decision.num_conditions should equal conditions.len()"
            );

            match num_conditions {
                0 => bug!("decision has no condition"),
                1..=ConditionId::MAX_AS_USIZE => (),
                _ => tcx.dcx().emit_warn(MCDCNumberConditionExceeded {
                    span: decision.span,
                    max_conditions: ConditionId::MAX_AS_USIZE,
                    num_conditions,
                }),
            }
        }
    }

    #[inline]
    fn increment_depth(&mut self) {
        self.decision_builder_stack.push(MCDCDecisionBuilder::default());
    }

    #[inline]
    fn decrement_depth(&mut self) {
        let Some(top_builder) = self.decision_builder_stack.pop() else {
            bug!("Unexpected empty decision builder stack");
        };
        if !top_builder.is_empty() {
            bug!("Popped a decision builder with unfinished decision in it")
        }
    }
}

impl Builder<'_, '_> {
    #[inline]
    fn mcdc_info_as_mut(&mut self) -> Option<&mut MCDCInfoBuilder> {
        self.coverage_info.as_mut().and_then(|cov_info| cov_info.mcdc_info.as_mut())
    }

    pub(crate) fn visit_logical_operator(&mut self, logical_op: LogicalOp, span: Span) {
        if let Some(mcdc_builder) = self.mcdc_info_as_mut() {
            mcdc_builder.record_operator(logical_op, span);
        }
    }
}
