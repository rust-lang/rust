use clippy_utils::diagnostics::span_lint;
use clippy_utils::is_adjusted;
use rustc_hir::{Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::declare_lint_pass;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for construction of a structure or tuple just to
    /// assign a value in it.
    ///
    /// ### Why is this bad?
    /// Readability. If the structure is only created to be
    /// updated, why not write the structure you want in the first place?
    ///
    /// ### Example
    /// ```no_run
    /// (0, 0).0 = 1
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub TEMPORARY_ASSIGNMENT,
    complexity,
    "assignments to temporaries"
}

fn is_temporary(expr: &Expr<'_>) -> bool {
    matches!(&expr.kind, ExprKind::Struct(..) | ExprKind::Tup(..))
}

declare_lint_pass!(TemporaryAssignment => [TEMPORARY_ASSIGNMENT]);

impl<'tcx> LateLintPass<'tcx> for TemporaryAssignment {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if let ExprKind::Assign(target, ..) = &expr.kind {
            let mut base = target;
            while let ExprKind::Field(f, _) | ExprKind::Index(f, _, _) = &base.kind {
                base = f;
            }
            if is_temporary(base) && !is_adjusted(cx, base) {
                span_lint(cx, TEMPORARY_ASSIGNMENT, expr.span, "assignment to temporary");
            }
        }
    }
}
