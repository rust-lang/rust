use if_chain::if_chain;
use rustc::hir::{Expr, ExprKind};
use rustc::lint::{LateContext, LateLintPass, LintArray, LintPass};
use rustc::{declare_lint_pass, declare_tool_lint};
use syntax_pos::Span;

use crate::consts::{constant, Constant};
use crate::utils::{in_macro_or_desugar, is_direct_expn_of, span_help_and_lint};

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
        let mut is_debug_assert = false;
        if let Some(assert_span) = is_direct_expn_of(e.span, "assert") {
            if in_macro_or_desugar(assert_span) {
                return;
            }
            if let Some(debug_assert_span) = is_direct_expn_of(assert_span, "debug_assert") {
                if in_macro_or_desugar(debug_assert_span) {
                    return;
                }
                is_debug_assert = true;
            }
            if let ExprKind::Unary(_, ref lit) = e.node {
                if let Some((bool_const, _)) = constant(cx, cx.tables, lit) {
                    if let Constant::Bool(is_true) bool_const {
                        if is_true {
                            span_help_and_lint(
                                cx,
                                ASSERTIONS_ON_CONSTANTS,
                                e.span,
                                "`assert!(true)` will be optimized out by the compiler",
                                "remove it"
                            );
                        } else if !is_debug_assert {
                            span_help_and_lint(
                                cx,
                                ASSERTIONS_ON_CONSTANTS,
                                e.span,
                                "`assert!(false)` should probably be replaced",
                                "use `panic!()` or `unreachable!()`"
                            );
                        }
                    }
                }
            }
        }
    }
}
