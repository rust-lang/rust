use clippy_utils::consts::{constant, Constant};
use clippy_utils::diagnostics::span_lint_and_help;
use clippy_utils::macros::{find_assert_args, root_macro_call_first_node, PanicExpn};
use rustc_hir::Expr;
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::sym;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `assert!(true)` and `assert!(false)` calls.
    ///
    /// ### Why is this bad?
    /// Will be optimized out by the compiler or should probably be replaced by a
    /// `panic!()` or `unreachable!()`
    ///
    /// ### Example
    /// ```rust,ignore
    /// assert!(false)
    /// assert!(true)
    /// const B: bool = false;
    /// assert!(B)
    /// ```
    #[clippy::version = "1.34.0"]
    pub ASSERTIONS_ON_CONSTANTS,
    style,
    "`assert!(true)` / `assert!(false)` will be optimized out by the compiler, and should probably be replaced by a `panic!()` or `unreachable!()`"
}

declare_lint_pass!(AssertionsOnConstants => [ASSERTIONS_ON_CONSTANTS]);

impl<'tcx> LateLintPass<'tcx> for AssertionsOnConstants {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, e: &'tcx Expr<'_>) {
        let Some(macro_call) = root_macro_call_first_node(cx, e) else {
            return;
        };
        let is_debug = match cx.tcx.get_diagnostic_name(macro_call.def_id) {
            Some(sym::debug_assert_macro) => true,
            Some(sym::assert_macro) => false,
            _ => return,
        };
        let Some((condition, panic_expn)) = find_assert_args(cx, e, macro_call.expn) else {
            return;
        };
        let Some(Constant::Bool(val)) = constant(cx, cx.typeck_results(), condition) else {
            return;
        };
        if val {
            span_lint_and_help(
                cx,
                ASSERTIONS_ON_CONSTANTS,
                macro_call.span,
                &format!(
                    "`{}!(true)` will be optimized out by the compiler",
                    cx.tcx.item_name(macro_call.def_id)
                ),
                None,
                "remove it",
            );
        } else if !is_debug {
            let (assert_arg, panic_arg) = match panic_expn {
                PanicExpn::Empty => ("", ""),
                _ => (", ..", ".."),
            };
            span_lint_and_help(
                cx,
                ASSERTIONS_ON_CONSTANTS,
                macro_call.span,
                &format!("`assert!(false{assert_arg})` should probably be replaced"),
                None,
                &format!("use `panic!({panic_arg})` or `unreachable!({panic_arg})`"),
            );
        }
    }
}
