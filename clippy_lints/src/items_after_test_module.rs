use clippy_utils::{diagnostics::span_lint_and_help, is_in_cfg_test};
use rustc_hir::{HirId, ItemId, ItemKind, Mod};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::sym;

declare_clippy_lint! {
    /// ### What it does
    /// Triggers if an item is declared after the testing module marked with `#[cfg(test)]`.
    /// ### Why is this bad?
    /// Having items declared after the testing module is confusing and may lead to bad test coverage.
    /// ### Example
    /// ```rust
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
    /// ```rust
    /// fn my_function() {
    ///     // [...]
    /// }
    ///
    /// #[cfg(test)]
    /// mod tests {
    ///     // [...]
    /// }
    /// ```
    #[clippy::version = "1.70.0"]
    pub ITEMS_AFTER_TEST_MODULE,
    style,
    "An item was found after the testing module `tests`"
}

declare_lint_pass!(ItemsAfterTestModule => [ITEMS_AFTER_TEST_MODULE]);

impl LateLintPass<'_> for ItemsAfterTestModule {
    fn check_mod(&mut self, cx: &LateContext<'_>, _: &Mod<'_>, _: HirId) {
        let mut was_test_mod_visited = false;
        let mut test_mod_hash: Option<u128> = None;

        let hir = cx.tcx.hir();
        let items = hir.items().collect::<Vec<ItemId>>();

        for itid in &items {
            let item = hir.item(*itid);

            if_chain! {
            if was_test_mod_visited;
            if cx.sess().source_map().lookup_char_pos(item.span.lo()).file.name_hash
            == test_mod_hash.unwrap(); // Will never fail
            if !matches!(item.kind, ItemKind::Mod(_) | ItemKind::Macro(_, _));
            if !is_in_cfg_test(cx.tcx, itid.hir_id()); // The item isn't in the testing module itself

            if !in_external_macro(cx.sess(), item.span);
            then {
                span_lint_and_help(cx, ITEMS_AFTER_TEST_MODULE, item.span, "an item was found after the testing module", None, "move the item to before the testing module was defined");
            }};

            if matches!(item.kind, ItemKind::Mod(_)) {
                for attr in cx.tcx.get_attrs(item.owner_id.to_def_id(), sym::cfg) {
                    if_chain! {
                                if attr.has_name(sym::cfg);
                                if let Some(mitems) = attr.meta_item_list();
                                if let [mitem] = &*mitems;
                                if mitem.has_name(sym::test);
                                then {
                                    was_test_mod_visited = true;
                    test_mod_hash = Some(cx.sess().source_map().lookup_char_pos(item.span.lo()).file.name_hash);
                                }
                            }
                }
            }
        }
    }
}
