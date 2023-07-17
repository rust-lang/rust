use clippy_utils::diagnostics::span_lint_and_note;
use clippy_utils::{is_in_cfg_test, is_in_test_function};
use rustc_hir::intravisit::FnKind;
use rustc_hir::{Body, FnDecl};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::def_id::LocalDefId;
use rustc_span::Span;

declare_clippy_lint! {
    /// ### What it does
    /// Triggers when a testing function (marked with the `#[test]` attribute) isn't inside a testing module
    /// (marked with `#[cfg(test)]`).
    /// ### Why is this bad?
    /// The idiomatic (and more performant) way of writing tests is inside a testing module (flagged with `#[cfg(test)]`),
    /// having test functions outside of this module is confusing and may lead to them being "hidden".
    /// ### Example
    /// ```rust
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
    /// ```rust
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
        if_chain! {
            if !matches!(kind, FnKind::Closure);
            if is_in_test_function(cx.tcx, body.id().hir_id);
            if !is_in_cfg_test(cx.tcx, body.id().hir_id);
            then {
                span_lint_and_note(
                    cx,
                    TESTS_OUTSIDE_TEST_MODULE,
                    sp,
                    "this function marked with #[test] is outside a #[cfg(test)] module",
                    None,
                    "move it to a testing module marked with #[cfg(test)]",
                );
            }
        }
    }
}
