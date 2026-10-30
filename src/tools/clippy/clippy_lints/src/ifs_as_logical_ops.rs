use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::higher::If;
use clippy_utils::source::{SpanExt as _, walk_span_to_context};
use clippy_utils::sugg::Sugg;
use clippy_utils::{is_else_clause, peel_blocks};
use rustc_ast::LitKind;
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass, LintContext as _, declare_lint_pass};
use rustc_span::Span;

declare_clippy_lint! {
    /// ### What it does
    ///
    /// Warn about cases where `x && y` could be used in place of an if condition.
    ///
    /// ### Why is this bad?
    ///
    /// `x && y` is more standard as a construction, and makes it clearer that this is just an and.
    /// It is also less verbose.
    ///
    /// ### Example
    /// ```no_run
    /// # fn a() -> bool { false }
    /// fn b(b1: bool) -> bool {
    ///     if b1 { a() } else { false }
    /// }
    ///
    /// ```
    ///
    /// Could be written
    ///
    /// ```no_run
    /// # fn a() -> bool { false }
    /// fn b(b1: bool) -> bool {
    ///     b1 && a()
    /// }
    /// ```
    #[clippy::version = "1.101.0"]
    pub IFS_AS_LOGICAL_OPS,
    pedantic,
    "`if` conditions that can be rewritten as logical operators"
}

declare_lint_pass!(IfsAsLogicalOps => [IFS_AS_LOGICAL_OPS]);

impl<'tcx> LateLintPass<'tcx> for IfsAsLogicalOps {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, e: &'tcx Expr<'tcx>) {
        // Make sure the if block is not an if-let block.
        if let Some(If {cond: if_cond, then, r#else: Some(els)}) = If::hir(e)
            && let ExprKind::Block(then_block, _label) = then.kind
            // Check if the if-block has only a trailing expression
            && then_block.stmts.is_empty()
            && let Some(then_block_inner_expr) = then_block.expr
            // And that the else block consists of only the boolean 'false'.
            && let ExprKind::Block(else_block, _label) = els.kind
            && else_block.stmts.is_empty()
            && let Some(else_block_inner_expr) = else_block.expr
            && let ExprKind::Lit(lit) = else_block_inner_expr.kind
            && matches!(lit.node, LitKind::Bool(false))
            // We do not emit this lint if the expression diverges.
            && !cx.typeck_results().expr_ty(then_block_inner_expr).is_never()
            // Make sure that the expression is only in a single macro context
            && let ctxt = e.span.ctxt()
            && ctxt == then_block.span.ctxt()
            && ctxt == else_block.span.ctxt()
            && ctxt == else_block_inner_expr.span.ctxt()
            && ctxt == lit.span.ctxt()
            && !ctxt.in_external_macro(cx.sess().source_map())
        {
            // Do not lint if the statement is trivially a boolean.
            if let ExprKind::Lit(lit_ptr) = peel_blocks(then_block_inner_expr).kind
                && let LitKind::Bool(_) = lit_ptr.node
            {
                return;
            }

            if let Some(walked_then_block_inner) = walk_span_to_context(then_block_inner_expr.span, ctxt)
                && check_that_nested_spans_have_no_comments(then_block.span, walked_then_block_inner, cx)
                && check_that_nested_spans_have_no_comments(else_block.span, else_block_inner_expr.span, cx)
            {
                let mut applicability = if ctxt.is_root() {
                    Applicability::MachineApplicable
                } else {
                    Applicability::MaybeIncorrect
                };

                let mut sugg = Sugg::hir_with_context(cx, if_cond, ctxt, "_", &mut applicability);
                let rhs_sugg = Sugg::hir_with_context(cx, then_block_inner_expr, ctxt, "_", &mut applicability);

                sugg = sugg.and(&rhs_sugg);

                if is_else_clause(cx.tcx, e) {
                    sugg = sugg.blockify();
                }

                span_lint_and_sugg(
                    cx,
                    IFS_AS_LOGICAL_OPS,
                    e.span,
                    "`if` expression that could be written as a logical && expression",
                    "try",
                    sugg.to_string(),
                    applicability,
                );
            }
        }
    }
}

fn check_that_nested_spans_have_no_comments(outer_span: Span, inner_span: Span, cx: &LateContext<'_>) -> bool {
    (outer_span.lo()..inner_span.lo()).check_text(cx, |src| src.trim_end() == "{")
        && (inner_span.hi()..outer_span.hi()).check_text(cx, |src| src.trim_start() == "}")
}
