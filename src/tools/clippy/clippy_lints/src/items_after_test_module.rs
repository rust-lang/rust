use clippy_utils::diagnostics::span_lint_hir_and_then;
use clippy_utils::source::SpanRangeExt;
use clippy_utils::{fulfill_or_allowed, is_cfg_test, is_from_proc_macro};
use rustc_errors::{Applicability, SuggestionStyle};
use rustc_hir::{HirId, Item, ItemKind, Mod};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::declare_lint_pass;
use rustc_span::hygiene::AstPass;
use rustc_span::{ExpnKind, sym};

declare_clippy_lint! {
    /// ### What it does
    /// Triggers if an item is declared after the testing module marked with `#[cfg(test)]`.
    /// ### Why is this bad?
    /// Having items declared after the testing module is confusing and may lead to bad test coverage.
    /// ### Example
    /// ```no_run
    /// #[cfg(test)]
    /// mod tests {
    ///     // [...]
    /// }
    ///
    /// fn my_function() {
    ///     // [...]
    /// }
    /// ```
    /// Use instead:
    /// ```no_run
    /// fn my_function() {
    ///     // [...]
    /// }
    ///
    /// #[cfg(test)]
    /// mod tests {
    ///     // [...]
    /// }
    /// ```
    #[clippy::version = "1.71.0"]
    pub ITEMS_AFTER_TEST_MODULE,
    style,
    "An item was found after the testing module `tests`"
}

declare_lint_pass!(ItemsAfterTestModule => [ITEMS_AFTER_TEST_MODULE]);

fn cfg_test_module<'tcx>(cx: &LateContext<'tcx>, item: &Item<'tcx>) -> bool {
    if let ItemKind::Mod(_, test_mod) = item.kind
        && item.span.hi() == test_mod.spans.inner_span.hi()
        && is_cfg_test(cx.tcx, item.hir_id())
        && !item.span.from_expansion()
        && !is_from_proc_macro(cx, item)
    {
        true
    } else {
        false
    }
}

impl LateLintPass<'_> for ItemsAfterTestModule {
    fn check_mod(&mut self, cx: &LateContext<'_>, module: &Mod<'_>, _: HirId) {
        let mut items = module.item_ids.iter().map(|&id| cx.tcx.hir_item(id));

        let Some((mod_pos, test_mod)) = items.by_ref().enumerate().find(|(_, item)| cfg_test_module(cx, item)) else {
            return;
        };

        let after: Vec<_> = items
            .filter(|item| {
                // Ignore the generated test main function
                if let ItemKind::Fn { ident, .. } = item.kind
                    && ident.name == sym::main
                    && item.span.ctxt().outer_expn_data().kind == ExpnKind::AstPass(AstPass::TestHarness)
                {
                    false
                } else {
                    true
                }
            })
            .collect();

        if let Some(last) = after.last()
            && after.iter().all(|&item| {
                !matches!(item.kind, ItemKind::Mod(..)) && !item.span.from_expansion() && !is_from_proc_macro(cx, item)
            })
            && !fulfill_or_allowed(cx, ITEMS_AFTER_TEST_MODULE, after.iter().map(|item| item.hir_id()))
        {
            let def_spans: Vec<_> = std::iter::once(test_mod.owner_id)
                .chain(after.iter().map(|item| item.owner_id))
                .map(|id| cx.tcx.def_span(id))
                .collect();

            span_lint_hir_and_then(
                cx,
                ITEMS_AFTER_TEST_MODULE,
                test_mod.hir_id(),
                def_spans,
                "items after a test module",
                |diag| {
                    if let Some(prev) = mod_pos.checked_sub(1)
                        && let prev = cx.tcx.hir_item(module.item_ids[prev])
                        && let items_span = last.span.with_lo(test_mod.span.hi())
                        && let Some(items) = items_span.get_source_text(cx)
                    {
                        diag.multipart_suggestion_with_style(
                            "move the items to before the test module was defined",
                            vec![
                                (prev.span.shrink_to_hi(), items.to_owned()),
                                (items_span, String::new()),
                            ],
                            Applicability::MachineApplicable,
                            SuggestionStyle::HideCodeAlways,
                        );
                    }
                },
            );
        }
    }
}
