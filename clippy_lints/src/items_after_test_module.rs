use clippy_utils::{diagnostics::span_lint_and_help, is_from_proc_macro, is_in_cfg_test};
use rustc_hir::{HirId, ItemId, ItemKind, Mod};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::{sym, Span};

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
        let mut test_mod_span: Option<Span> = None;

        let hir = cx.tcx.hir();
        let items = hir.items().collect::<Vec<ItemId>>();

        for (i, itid) in items.iter().enumerate() {
            let item = hir.item(*itid);

            if_chain! {
            if was_test_mod_visited;
            if i == (items.len() - 3 /* Weird magic number (HIR-translation behaviour) */);
            if cx.sess().source_map().lookup_char_pos(item.span.lo()).file.name_hash
            == cx.sess().source_map().lookup_char_pos(test_mod_span.unwrap().lo()).file.name_hash; // Will never fail
            if !matches!(item.kind, ItemKind::Mod(_));
            if !is_in_cfg_test(cx.tcx, itid.hir_id()); // The item isn't in the testing module itself
            if !in_external_macro(cx.sess(), item.span);
            if !is_from_proc_macro(cx, item);

            then {
                span_lint_and_help(cx, ITEMS_AFTER_TEST_MODULE, test_mod_span.unwrap().with_hi(item.span.hi()), "items were found after the testing module", None, "move the items to before the testing module was defined");
            }};

            if let ItemKind::Mod(module) = item.kind && item.span.hi() == module.spans.inner_span.hi() {
			// Check that it works the same way, the only I way I've found for #10713
				for attr in cx.tcx.get_attrs(item.owner_id.to_def_id(), sym::cfg) {
					if_chain! {
						if attr.has_name(sym::cfg);
                        if let Some(mitems) = attr.meta_item_list();
                        if let [mitem] = &*mitems;
                        if mitem.has_name(sym::test);
                        then {
							was_test_mod_visited = true;
                            test_mod_span = Some(item.span);
                        }
                    }
                }
			}
        }
    }
}
