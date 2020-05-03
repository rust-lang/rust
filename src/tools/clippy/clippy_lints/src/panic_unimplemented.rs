use crate::utils::{is_expn_of, match_panic_call, span_lint};
use if_chain::if_chain;
use rustc_hir::Expr;
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::Span;

declare_clippy_lint! {
    /// **What it does:** Checks for usage of `panic!`.
    ///
    /// **Why is this bad?** `panic!` will stop the execution of the executable
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```no_run
    /// panic!("even with a good reason");
    /// ```
    pub PANIC,
    restriction,
    "usage of the `panic!` macro"
}

declare_clippy_lint! {
    /// **What it does:** Checks for usage of `unimplemented!`.
    ///
    /// **Why is this bad?** This macro should not be present in production code
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```no_run
    /// unimplemented!();
    /// ```
    pub UNIMPLEMENTED,
    restriction,
    "`unimplemented!` should not be present in production code"
}

declare_clippy_lint! {
    /// **What it does:** Checks for usage of `todo!`.
    ///
    /// **Why is this bad?** This macro should not be present in production code
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```no_run
    /// todo!();
    /// ```
    pub TODO,
    restriction,
    "`todo!` should not be present in production code"
}

declare_clippy_lint! {
    /// **What it does:** Checks for usage of `unreachable!`.
    ///
    /// **Why is this bad?** This macro can cause code to panic
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```no_run
    /// unreachable!();
    /// ```
    pub UNREACHABLE,
    restriction,
    "usage of the `unreachable!` macro"
}

declare_lint_pass!(PanicUnimplemented => [UNIMPLEMENTED, UNREACHABLE, TODO, PANIC]);

impl<'tcx> LateLintPass<'tcx> for PanicUnimplemented {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if match_panic_call(cx, expr).is_some() {
            let span = get_outer_span(cx, expr);
            let expr_span = cx.tcx.hir().span(expr.hir_id);
            if is_expn_of(expr_span, "unimplemented").is_some() {
                span_lint(
                    cx,
                    UNIMPLEMENTED,
                    span,
                    "`unimplemented` should not be present in production code",
                );
            } else if is_expn_of(expr_span, "todo").is_some() {
                span_lint(cx, TODO, span, "`todo` should not be present in production code");
            } else if is_expn_of(expr_span, "unreachable").is_some() {
                span_lint(cx, UNREACHABLE, span, "usage of the `unreachable!` macro");
            } else if is_expn_of(expr_span, "panic").is_some() {
                span_lint(cx, PANIC, span, "`panic` should not be present in production code");
            }
        }
    }
}

fn get_outer_span(cx: &LateContext<'_>, expr: &Expr<'_>) -> Span {
    let expr_span = cx.tcx.hir().span(expr.hir_id);
    if_chain! {
        if expr_span.from_expansion();
        let first = expr_span.ctxt().outer_expn_data();
        if first.call_site.from_expansion();
        let second = first.call_site.ctxt().outer_expn_data();
        then {
            second.call_site
        } else {
            expr_span
        }
    }
}
