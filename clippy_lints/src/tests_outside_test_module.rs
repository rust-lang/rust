use clippy_utils::{diagnostics::span_lint_and_note, is_in_cfg_test, is_in_test_function, is_test_module_or_function};
use rustc_data_structures::sync::par_for_each_in;
use rustc_hir::{intravisit::FnKind, Body, FnDecl, HirId, ItemKind, Mod};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::{def_id::LocalDefId, Span};

declare_clippy_lint! {
    /// ### What it does
    ///
    /// Triggers when a testing function (marked with the `#[test]` attribute) isn't inside a testing module (marked with `#[cfg(test)]`).
    ///
    /// ### Why is this bad?
    ///
    /// The idiomatic (and more performant) way of writing tests is inside a testing module (flagged with `#[cfg(test)]`), having test functions outside of this module is confusing and may lead to them being "hidden".
    ///
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
    "The test function `my_cool_test` is outside the testing module `tests`."
}

pub(crate) struct TestsOutsideTestModule {
    pub test_mod_exists: bool,
}

impl TestsOutsideTestModule {
    pub fn new() -> Self {
        Self { test_mod_exists: false }
    }
}

impl_lint_pass!(TestsOutsideTestModule => [TESTS_OUTSIDE_TEST_MODULE]);

impl LateLintPass<'_> for TestsOutsideTestModule {
    fn check_mod(&mut self, cx: &LateContext<'_>, _: &Mod<'_>, _: HirId) {
        self.test_mod_exists = false;

        // par_for_each_item uses Fn, while par_for_each_in uses FnMut
        par_for_each_in(cx.tcx.hir_crate_items(()).items(), |itemid| {
            let item = cx.tcx.hir().item(itemid);
            if_chain! {
                if matches!(item.kind, ItemKind::Mod(_));
                if is_test_module_or_function(cx.tcx, item);
                then {
                    self.test_mod_exists = true;
                }
            }
        });
    }

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
            if self.test_mod_exists;
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
