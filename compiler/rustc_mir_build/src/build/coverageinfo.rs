use std::assert_matches::assert_matches;
use std::collections::hash_map::Entry;

use rustc_data_structures::fx::FxHashMap;
use rustc_index::IndexVec;
use rustc_middle::mir::coverage::{
    BlockMarkerId, BranchSpan, CoverageKind, DecisionId, DecisionSpan,
};
use rustc_middle::mir::{self, BasicBlock, UnOp};
use rustc_middle::thir::{ExprId, ExprKind, Thir};
use rustc_middle::ty::TyCtxt;
use rustc_span::def_id::LocalDefId;
use rustc_span::Span;

use crate::build::Builder;
use crate::errors::MCDCNestedDecision;

pub(crate) struct BranchInfoBuilder {
    /// Maps condition expressions to their enclosing `!`, for better instrumentation.
    nots: FxHashMap<ExprId, NotInfo>,

    num_block_markers: usize,
    branch_spans: Vec<BranchSpan>,

    // MCDC decision stuff
    /// ID of the current decision.
    /// Do not use directly. Use the function instead, as it will hide
    /// the decision in the scope of nested decisions.
    current_decision_id: Option<DecisionId>,
    /// Track the nesting level of decision to avoid MCDC instrumentation of
    /// nested decisions.
    nested_decision_level: u32,
    /// Vector for storing all the decisions with their span
    decisions: IndexVec<DecisionId, Span>,
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
        if (tcx.sess.instrument_coverage_branch() || tcx.sess.instrument_coverage_mcdc())
            && tcx.is_eligible_for_coverage(def_id)
        {
            Some(Self {
                nots: FxHashMap::default(),
                num_block_markers: 0,
                branch_spans: vec![],
                current_decision_id: None,
                nested_decision_level: 0,
                decisions: IndexVec::new(),
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

    fn next_block_marker_id(&mut self) -> BlockMarkerId {
        let id = BlockMarkerId::from_usize(self.num_block_markers);
        self.num_block_markers += 1;
        id
    }

    pub(crate) fn into_done(self) -> Option<Box<mir::coverage::BranchInfo>> {
        let Self { nots: _, num_block_markers, branch_spans, decisions, .. } = self;

        if num_block_markers == 0 {
            assert!(branch_spans.is_empty());
            return None;
        }

        let mut decision_spans = IndexVec::from_iter(
            decisions.into_iter().map(|span| DecisionSpan { span, num_conditions: 0 }),
        );

        // Count the number of conditions linked to each decision.
        if !decision_spans.is_empty() {
            for branch_span in branch_spans.iter() {
                decision_spans[branch_span.decision_id].num_conditions += 1;
            }
        }

        Some(Box::new(mir::coverage::BranchInfo {
            num_block_markers,
            branch_spans,
            decision_spans,
        }))
    }

    /// Increase the nested decision level and return true if the
    /// decision can be instrumented (not in a nested condition).
    pub fn enter_decision(&mut self, span: Span) -> bool {
        self.nested_decision_level += 1;
        let can_mcdc = !self.in_nested_condition();

        if can_mcdc {
            self.current_decision_id = Some(self.decisions.push(span));
        }

        can_mcdc
    }

    pub fn exit_decision(&mut self) {
        self.nested_decision_level -= 1;
    }

    /// Return true if the current decision is located inside another decision.
    pub fn in_nested_condition(&self) -> bool {
        self.nested_decision_level > 1
    }

    pub fn current_decision_id(&self) -> Option<DecisionId> {
        if self.in_nested_condition() { None } else { self.current_decision_id }
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
        let Some(branch_info) = self.coverage_branch_info.as_ref() else { return };

        // If this condition expression is nested within one or more `!` expressions,
        // replace it with the enclosing `!` collected by `visit_unary_not`.
        if let Some(&NotInfo { enclosing_not, is_flipped }) = branch_info.nots.get(&expr_id) {
            expr_id = enclosing_not;
            if is_flipped {
                std::mem::swap(&mut then_block, &mut else_block);
            }
        }
        let source_info = self.source_info(self.thir[expr_id].span);

        // Now that we have `source_info`, we can upgrade to a &mut reference.
        let branch_info = self.coverage_branch_info.as_mut().expect("upgrading & to &mut");

        let mut inject_branch_marker = |block: BasicBlock| {
            let id = branch_info.next_block_marker_id();

            let marker_statement = mir::Statement {
                source_info,
                kind: mir::StatementKind::Coverage(CoverageKind::BlockMarker { id }),
            };
            self.cfg.push(block, marker_statement);

            id
        };

        let true_marker = inject_branch_marker(then_block);
        let false_marker = inject_branch_marker(else_block);

        branch_info.branch_spans.push(BranchSpan {
            span: source_info.span,
            // FIXME(dprn): Handle case when MCDC is disabled better than just putting 0.
            decision_id: branch_info.current_decision_id.unwrap_or(DecisionId::from_u32(0)),
            true_marker,
            false_marker,
        });
    }

    /// If MCDC coverage is enabled, inject a decision entry marker in the given decision.
    /// return true
    pub(crate) fn begin_mcdc_decision_coverage(&mut self, expr_id: ExprId, block: BasicBlock) {
        // Early return if MCDC coverage is not enabled.
        if !self.tcx.sess.instrument_coverage_mcdc() {
            return;
        }
        let Some(branch_info) = self.coverage_branch_info.as_mut() else {
            return;
        };

        let span = self.thir[expr_id].span;

        // enter_decision returns false if it detects nested decisions.
        if !branch_info.enter_decision(span) {
            // FIXME(dprn): do WARNING for nested decision.
            debug!("MCDC: Unsupported nested decision");
            self.tcx.dcx().emit_warn(MCDCNestedDecision { span });
            return;
        }

        let decision_id = branch_info.current_decision_id().expect("Should have returned.");

        // Inject a decision marker
        let source_info = self.source_info(span);
        let marker_statement = mir::Statement {
            source_info,
            kind: mir::StatementKind::Coverage(CoverageKind::MCDCDecisionEntryMarker {
                id: decision_id,
            }),
        };
        self.cfg.push(block, marker_statement);
    }

    /// If MCDC is enabled, and function is instrumented,
    pub(crate) fn end_mcdc_decision_coverage(&mut self) {
        // Early return if MCDC coverage is not enabled.
        if !self.tcx.sess.instrument_coverage_mcdc() {
            return;
        }
        let Some(branch_info) = self.coverage_branch_info.as_mut() else {
            return;
        };

        // Exit decision now so we can drop &mut to branch info
        branch_info.exit_decision();
    }

    /// If MCDC is enabled and the current decision is being instrumented,
    /// inject an `MCDCDecisionOutputMarker` to the given basic block.
    /// `outcome` should be true for the then block and false for the else block.
    pub(crate) fn mcdc_decision_outcome_block(
        &mut self,
        bb: BasicBlock,
        outcome: bool,
    ) -> BasicBlock {
        let Some(branch_info) = self.coverage_branch_info.as_mut() else {
            // Coverage instrumentation is not enabled.
            return bb;
        };
        let Some(decision_id) = branch_info.current_decision_id() else {
            // Decision is not instrumented
            return bb;
        };

        let span = branch_info.decisions[decision_id];
        let source_info = self.source_info(span);
        let marker_statement = mir::Statement {
            source_info,
            kind: mir::StatementKind::Coverage(CoverageKind::MCDCDecisionOutputMarker {
                id: decision_id,
                outcome,
            }),
        };

        // Insert statements at the beginning of the following basic block
        self.cfg.block_data_mut(bb).statements.insert(0, marker_statement);

        // Create a new block to return
        let new_bb = self.cfg.start_new_block();

        // Set bb -> new_bb
        self.cfg.goto(bb, source_info, new_bb);

        new_bb
    }
}
