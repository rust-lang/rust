use super::FOR_UNBOUNDED_RANGE;
use super::infinite_loop::LoopVisitor;
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::higher;
use rustc_ast::Label;
use rustc_hir::Expr;
use rustc_hir::intravisit::Visitor as _;
use rustc_lint::LateContext;
use rustc_span::Span;

pub fn check<'tcx>(
    cx: &LateContext<'tcx>,
    label: Option<Label>,
    arg: &'tcx Expr<'tcx>,
    span: Span,
    body: &'tcx Expr<'tcx>,
) {
    if let Some(range) = higher::Range::hir(cx, arg)
        && let Some(range_start) = range.start
        && let None = range.end
        && let ty = cx.typeck_results.expr_ty_adjusted(range_start)
        && (ty.is_integral() || ty.is_char())
    {
        let mut loop_visitor = LoopVisitor::new(cx, label);

        loop_visitor.visit_expr(body);

        if loop_visitor.is_finite {
            // The loop is likely finite.
            return;
        }

        span_lint_and_then(cx, FOR_UNBOUNDED_RANGE, span, "for loop on unbounded range", |diag| {
            diag.span_suggestion_verbose(
                arg.span.shrink_to_hi(),
                "for loops over unbounded ranges will wrap around and may panic, consider adding an inclusive high bound",
                format!("={ty}::MAX"),
                rustc_errors::Applicability::MaybeIncorrect,
            );
        });
    }
}
