use if_chain::if_chain;
use rustc::hir::{Expr, ExprKind};
use rustc::lint::{LateContext, LateLintPass, LintArray, LintPass};
use rustc::{declare_tool_lint, lint_array};

use crate::consts::{constant, Constant};
use crate::syntax::ast::LitKind;
use crate::utils::{in_macro, is_direct_expn_of, span_help_and_lint};

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

pub struct AssertionsOnConstants;

impl LintPass for AssertionsOnConstants {
    fn get_lints(&self) -> LintArray {
        lint_array![ASSERTIONS_ON_CONSTANTS]
    }

    fn name(&self) -> &'static str {
        "AssertionsOnConstants"
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for AssertionsOnConstants {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, e: &'tcx Expr) {
        if_chain! {
            if let Some(assert_span) = is_direct_expn_of(e.span, "assert");
            if !in_macro(assert_span)
                || is_direct_expn_of(assert_span, "debug_assert").map_or(false, |span| !in_macro(span));
            if let ExprKind::Unary(_, ref lit) = e.node;
            then {
                if let ExprKind::Lit(ref inner) = lit.node {
                    match inner.node {
                        LitKind::Bool(true) => {
                            span_help_and_lint(cx, ASSERTIONS_ON_CONSTANTS, e.span,
                                "assert!(true) will be optimized out by the compiler",
                                "remove it");
                        },
                        LitKind::Bool(false) => {
                            span_help_and_lint(
                                cx, ASSERTIONS_ON_CONSTANTS, e.span,
                                "assert!(false) should probably be replaced",
                                "use panic!() or unreachable!()");
                        },
                        _ => (),
                    }
                } else if let Some(bool_const) = constant(cx, cx.tables, lit) {
                    match bool_const.0 {
                        Constant::Bool(true) => {
                            span_help_and_lint(cx, ASSERTIONS_ON_CONSTANTS, e.span,
                                "assert!(const: true) will be optimized out by the compiler",
                                "remove it");
                        },
                        Constant::Bool(false) => {
                            span_help_and_lint(cx, ASSERTIONS_ON_CONSTANTS, e.span,
                                "assert!(const: false) should probably be replaced",
                                "use panic!() or unreachable!()");
                        },
                        _ => (),
                    }
                }
            }
        }
    }
}
