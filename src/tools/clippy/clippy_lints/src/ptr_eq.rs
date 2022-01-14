use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet_opt;
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir::{BinOpKind, Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// ### What it does
    /// Use `std::ptr::eq` when applicable
    ///
    /// ### Why is this bad?
    /// `ptr::eq` can be used to compare `&T` references
    /// (which coerce to `*const T` implicitly) by their address rather than
    /// comparing the values they point to.
    ///
    /// ### Example
    /// ```rust
    /// let a = &[1, 2, 3];
    /// let b = &[1, 2, 3];
    ///
    /// assert!(a as *const _ as usize == b as *const _ as usize);
    /// ```
    /// Use instead:
    /// ```rust
    /// let a = &[1, 2, 3];
    /// let b = &[1, 2, 3];
    ///
    /// assert!(std::ptr::eq(a, b));
    /// ```
    #[clippy::version = "1.49.0"]
    pub PTR_EQ,
    style,
    "use `std::ptr::eq` when comparing raw pointers"
}

declare_lint_pass!(PtrEq => [PTR_EQ]);

static LINT_MSG: &str = "use `std::ptr::eq` when comparing raw pointers";

impl<'tcx> LateLintPass<'tcx> for PtrEq {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if expr.span.from_expansion() {
            return;
        }

        if let ExprKind::Binary(ref op, left, right) = expr.kind {
            if BinOpKind::Eq == op.node {
                let (left, right) = match (expr_as_cast_to_usize(cx, left), expr_as_cast_to_usize(cx, right)) {
                    (Some(lhs), Some(rhs)) => (lhs, rhs),
                    _ => (left, right),
                };

                if_chain! {
                    if let Some(left_var) = expr_as_cast_to_raw_pointer(cx, left);
                    if let Some(right_var) = expr_as_cast_to_raw_pointer(cx, right);
                    if let Some(left_snip) = snippet_opt(cx, left_var.span);
                    if let Some(right_snip) = snippet_opt(cx, right_var.span);
                    then {
                        span_lint_and_sugg(
                            cx,
                            PTR_EQ,
                            expr.span,
                            LINT_MSG,
                            "try",
                            format!("std::ptr::eq({}, {})", left_snip, right_snip),
                            Applicability::MachineApplicable,
                            );
                    }
                }
            }
        }
    }
}

// If the given expression is a cast to a usize, return the lhs of the cast
// E.g., `foo as *const _ as usize` returns `foo as *const _`.
fn expr_as_cast_to_usize<'tcx>(cx: &LateContext<'tcx>, cast_expr: &'tcx Expr<'_>) -> Option<&'tcx Expr<'tcx>> {
    if cx.typeck_results().expr_ty(cast_expr) == cx.tcx.types.usize {
        if let ExprKind::Cast(expr, _) = cast_expr.kind {
            return Some(expr);
        }
    }
    None
}

// If the given expression is a cast to a `*const` pointer, return the lhs of the cast
// E.g., `foo as *const _` returns `foo`.
fn expr_as_cast_to_raw_pointer<'tcx>(cx: &LateContext<'tcx>, cast_expr: &'tcx Expr<'_>) -> Option<&'tcx Expr<'tcx>> {
    if cx.typeck_results().expr_ty(cast_expr).is_unsafe_ptr() {
        if let ExprKind::Cast(expr, _) = cast_expr.kind {
            return Some(expr);
        }
    }
    None
}
