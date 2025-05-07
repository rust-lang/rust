use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::is_in_test;
use clippy_utils::macros::{PanicExpn, find_assert_args, find_assert_eq_args, root_macro_call_first_node};
use rustc_hir::Expr;
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::declare_lint_pass;
use rustc_span::sym;

declare_clippy_lint! {
    /// ### What it does
    /// Checks assertions without a custom panic message.
    ///
    /// ### Why restrict this?
    /// Without a good custom message, it'd be hard to understand what went wrong when the assertion fails.
    /// A good custom message should be more about why the failure of the assertion is problematic
    /// and not what is failed because the assertion already conveys that.
    ///
    /// Although the same reasoning applies to testing functions, this lint ignores them as they would be too noisy.
    /// Also, in most cases understanding the test failure would be easier
    /// compared to understanding a complex invariant distributed around the codebase.
    ///
    /// ### Known problems
    /// This lint cannot check the quality of the custom panic messages.
    /// Hence, you can suppress this lint simply by adding placeholder messages
    /// like "assertion failed". However, we recommend coming up with good messages
    /// that provide useful information instead of placeholder messages that
    /// don't provide any extra information.
    ///
    /// ### Example
    /// ```no_run
    /// # struct Service { ready: bool }
    /// fn call(service: Service) {
    ///     assert!(service.ready);
    /// }
    /// ```
    /// Use instead:
    /// ```no_run
    /// # struct Service { ready: bool }
    /// fn call(service: Service) {
    ///     assert!(service.ready, "`service.poll_ready()` must be called first to ensure that service is ready to receive requests");
    /// }
    /// ```
    #[clippy::version = "1.70.0"]
    pub MISSING_ASSERT_MESSAGE,
    restriction,
    "checks assertions without a custom panic message"
}

declare_lint_pass!(MissingAssertMessage => [MISSING_ASSERT_MESSAGE]);

impl<'tcx> LateLintPass<'tcx> for MissingAssertMessage {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        let Some(macro_call) = root_macro_call_first_node(cx, expr) else {
            return;
        };
        let single_argument = match cx.tcx.get_diagnostic_name(macro_call.def_id) {
            Some(sym::assert_macro | sym::debug_assert_macro) => true,
            Some(
                sym::assert_eq_macro | sym::assert_ne_macro | sym::debug_assert_eq_macro | sym::debug_assert_ne_macro,
            ) => false,
            _ => return,
        };

        // This lint would be very noisy in tests, so just ignore if we're in test context
        if is_in_test(cx.tcx, expr.hir_id) {
            return;
        }

        let panic_expn = if single_argument {
            let Some((_, panic_expn)) = find_assert_args(cx, expr, macro_call.expn) else {
                return;
            };
            panic_expn
        } else {
            let Some((_, _, panic_expn)) = find_assert_eq_args(cx, expr, macro_call.expn) else {
                return;
            };
            panic_expn
        };

        if let PanicExpn::Empty = panic_expn {
            #[expect(clippy::collapsible_span_lint_calls, reason = "rust-clippy#7797")]
            span_lint_and_then(
                cx,
                MISSING_ASSERT_MESSAGE,
                macro_call.span,
                "assert without any message",
                |diag| {
                    diag.help("consider describing why the failing assert is problematic");
                },
            );
        }
    }
}
