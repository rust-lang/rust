use rustc::hir::{Expr, ExprKind};
use rustc::lint::{LateContext, LateLintPass, LintArray, LintPass};
use rustc::{declare_lint_pass, declare_tool_lint};

use crate::consts::{constant, Constant};
use crate::utils::{is_direct_expn_of, is_expn_of, span_help_and_lint};

declare_clippy_lint! {
    /// **What it does:** Checks for `assert!(true)` and `assert!(false)` calls.
    ///
    /// **Why is this bad?** Will be optimized out by the compiler or should probably be replaced by a
    /// panic!() or unreachable!()
    ///
    /// **Known problems:** None
    ///
    /// **Example:**
    /// ```rust,ignore
    /// assert!(false)
    /// // or
    /// assert!(true)
    /// // or
    /// const B: bool = false;
    /// assert!(B)
    /// ```
    pub ASSERTIONS_ON_CONSTANTS,
    style,
    "`assert!(true)` / `assert!(false)` will be optimized out by the compiler, and should probably be replaced by a `panic!()` or `unreachable!()`"
}

declare_lint_pass!(AssertionsOnConstants => [ASSERTIONS_ON_CONSTANTS]);

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for AssertionsOnConstants {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, e: &'tcx Expr) {
        let lint_assert_cb = |is_debug_assert: bool| {
            if let ExprKind::Unary(_, ref lit) = e.node {
                if let Some((Constant::Bool(is_true), _)) = constant(cx, cx.tables, lit) {
                    if is_true {
                        span_help_and_lint(
                            cx,
                            ASSERTIONS_ON_CONSTANTS,
                            e.span,
                            "`assert!(true)` will be optimized out by the compiler",
                            "remove it",
                        );
                    } else if !is_debug_assert {
                        span_help_and_lint(
                            cx,
                            ASSERTIONS_ON_CONSTANTS,
                            e.span,
                            "`assert!(false)` should probably be replaced",
                            "use `panic!()` or `unreachable!()`",
                        );
                    }
                }
            }
        };
        if let Some(debug_assert_span) = is_expn_of(e.span, "debug_assert") {
            if debug_assert_span.from_expansion() {
                return;
            }
            lint_assert_cb(true);
        } else if let Some(assert_span) = is_direct_expn_of(e.span, "assert") {
            if assert_span.from_expansion() {
                return;
            }
            lint_assert_cb(false);
        }
    }
}
