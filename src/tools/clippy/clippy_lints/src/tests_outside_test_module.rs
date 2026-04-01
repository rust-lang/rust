use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::{is_in_cfg_test, is_in_test_function};
use rustc_hir::intravisit::FnKind;
use rustc_hir::{Body, FnDecl};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::declare_lint_pass;
use rustc_span::Span;
use rustc_span::def_id::LocalDefId;

declare_clippy_lint! {
    /// ### What it does
    /// Triggers when a testing function (marked with the `#[test]` attribute) isn't inside a testing module
    /// (marked with `#[cfg(test)]`).
    ///
    /// ### Why restrict this?
    /// The idiomatic (and more performant) way of writing tests is inside a testing module (flagged with `#[cfg(test)]`),
    /// having test functions outside of this module is confusing and may lead to them being "hidden".
    ///
    /// ### Example
    /// ```no_run
    /// #[test]
    /// fn my_cool_test() {
    ///     // [...]
    /// }
    ///
    /// #[cfg(test)]
    /// mod tests {
    ///     // [...]
    /// }
    ///
    /// ```
    /// Use instead:
    /// ```no_run
    /// #[cfg(test)]
    /// mod tests {
    ///     #[test]
    ///     fn my_cool_test() {
    ///         // [...]
    ///     }
    /// }
    /// ```
    #[clippy::version = "1.70.0"]
    pub TESTS_OUTSIDE_TEST_MODULE,
    restriction,
    "A test function is outside the testing module."
}

declare_lint_pass!(TestsOutsideTestModule => [TESTS_OUTSIDE_TEST_MODULE]);

impl LateLintPass<'_> for TestsOutsideTestModule {
    fn check_fn(
        &mut self,
        cx: &LateContext<'_>,
        kind: FnKind<'_>,
        _: &FnDecl<'_>,
        body: &Body<'_>,
        sp: Span,
        _: LocalDefId,
    ) {
        if !matches!(kind, FnKind::Closure)
            && is_in_test_function(cx.tcx, body.id().hir_id)
            && !is_in_cfg_test(cx.tcx, body.id().hir_id)
        {
            #[expect(clippy::collapsible_span_lint_calls, reason = "rust-clippy#7797")]
            span_lint_and_then(
                cx,
                TESTS_OUTSIDE_TEST_MODULE,
                sp,
                "this function marked with #[test] is outside a #[cfg(test)] module",
                |diag| {
                    diag.note("move it to a testing module marked with #[cfg(test)]");
                },
            );
        }
    }
}
