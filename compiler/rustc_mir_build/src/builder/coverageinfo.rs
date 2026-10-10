use rustc_hir::HirId;
use rustc_middle::mir::coverage::{CoverageKind, PointKind};
use rustc_middle::mir::{self, BasicBlock, SourceInfo, Statement};
use rustc_middle::thir;

use crate::builder::Builder;

impl<'tcx> Builder<'_, 'tcx> {
    /// Does nothing if `-Cinstrument-coverage` is not enabled.
    ///
    /// Otherwise, pushes a marker statement to `block` indicating that this is where
    /// the HIR expression `hir_id` is being evaluated.
    pub(crate) fn push_coverage_point_for_expr(
        &mut self,
        block: BasicBlock,
        source_info: SourceInfo,
        hir_id: HirId,
    ) {
        if !self.tcx.sess.instrument_coverage() {
            return;
        }
        self.push_coverage_point_inner(block, source_info, PointKind::Expr, hir_id);
    }

    /// Does nothing if `-Cinstrument-coverage` is not enabled.
    ///
    /// Otherwise, pushes a marker statement to `block` indicating that this is where
    /// the one-sided if-expression `if_expr` will generate its synthetic `else {}`
    /// path, since it lacks an explicit `else` block.
    pub(crate) fn push_coverage_point_for_implicit_else(
        &mut self,
        block: BasicBlock,
        source_info: SourceInfo,
        if_expr: &thir::Expr<'tcx>,
    ) {
        if !self.tcx.sess.instrument_coverage() {
            return;
        }
        let hir_id = self.recover_hir_id_for_expr(if_expr);
        self.push_coverage_point_inner(block, source_info, PointKind::ImplicitElse, hir_id);
    }

    /// Does nothing if `-Cinstrument-coverage` is not enabled.
    ///
    /// Otherwise, pushes a marker statement to `block` indicating that this is where
    /// the function `fn_hir_id` would implicitly return at the end of its body.
    pub(crate) fn push_coverage_point_for_fn_end(
        &mut self,
        block: BasicBlock,
        source_info: SourceInfo,
        fn_hir_id: HirId,
    ) {
        if !self.tcx.sess.instrument_coverage() {
            return;
        }
        self.push_coverage_point_inner(block, source_info, PointKind::FunctionEnd, fn_hir_id);
    }

    /// Does nothing if branch coverage is not enabled.
    ///
    /// Otherwise, pushes marker statements to `true_block` and `false_block`
    /// indicating that a branch to one of those blocks occurred due to inspection
    /// of `scrutinee_expr`.
    pub(crate) fn push_coverage_points_for_branch_outcomes(
        &mut self,
        scrutinee_expr: &thir::Expr<'tcx>,
        true_block: BasicBlock,
        false_block: BasicBlock,
    ) {
        if !self.tcx.sess.instrument_coverage_branch() {
            return;
        }

        let hir_id = self.recover_hir_id_for_expr(scrutinee_expr);
        let source_info = self.source_info(scrutinee_expr.span);
        let pk_branch_outcome = |outcome: bool| PointKind::BranchOutcome { outcome };
        self.push_coverage_point_inner(true_block, source_info, pk_branch_outcome(true), hir_id);
        self.push_coverage_point_inner(false_block, source_info, pk_branch_outcome(false), hir_id);
    }

    /// Recovers the full [`HirId`] for a THIR expression by combining its local ID
    /// with the current function's owner ID.
    fn recover_hir_id_for_expr(&self, expr: &thir::Expr<'tcx>) -> HirId {
        // Note that we can't call `hir_id.expect_owner()`, because it would fail
        // if we're inside a closure, for example.
        HirId { owner: self.hir_id.owner, local_id: expr.temp_scope_id }
    }

    fn push_coverage_point_inner(
        &mut self,
        block: BasicBlock,
        source_info: SourceInfo,
        point_kind: PointKind,
        hir_id: HirId,
    ) {
        assert!(self.tcx.sess.instrument_coverage());

        let stmt = Statement::new(
            source_info,
            mir::StatementKind::Coverage(CoverageKind::Point { point_kind, hir_id }),
        );
        self.cfg.push(block, stmt);
    }

    /// If condition coverage is enabled, inject extra blocks and marker statements
    /// that will let us track the value of the condition in `place`.
    pub(crate) fn visit_coverage_standalone_condition(
        &mut self,
        expr_id: thir::ExprId,   // Expression being inspected
        place: mir::Place<'tcx>, // Already holds the boolean condition value
        block: &mut BasicBlock,
    ) {
        // Bail out if condition coverage is not enabled for this function.
        if !self.tcx.sess.instrument_coverage_condition() {
            return;
        };

        // Remove any wrappers, so that we can inspect the real underlying expression.
        let mut expr = &self.thir[expr_id];
        while let thir::ExprKind::ValueExpr { source: inner }
        | thir::ExprKind::Scope { value: inner, .. } = expr.kind
        {
            expr = &self.thir[inner];
        }
        // If the expression is a lazy logical op, it will naturally get branch
        // coverage as part of its normal lowering, so we can disregard it here.
        if let thir::ExprKind::LogicalOp { .. } = expr.kind {
            return;
        }

        // Using the boolean value that has already been stored in `place`, set up
        // control flow in the shape of a diamond, so that we can place separate
        // marker statements in the true and false blocks. The coverage MIR pass
        // will use those markers to inject coverage counters as appropriate.
        //
        //          block
        //         /     \
        // true_block   false_block
        //  (marker)     (marker)
        //         \     /
        //        join_block

        let source_info = self.source_info(expr.span);
        let true_block = self.cfg.start_new_block();
        let false_block = self.cfg.start_new_block();
        self.cfg.terminate(
            *block,
            source_info,
            mir::TerminatorKind::if_(mir::Operand::Copy(place), true_block, false_block),
        );

        self.push_coverage_points_for_branch_outcomes(expr, true_block, false_block);

        let join_block = self.cfg.start_new_block();
        self.cfg.goto(true_block, source_info, join_block);
        self.cfg.goto(false_block, source_info, join_block);
        // Any subsequent codegen in the caller should use the new join block.
        *block = join_block;
    }
}
