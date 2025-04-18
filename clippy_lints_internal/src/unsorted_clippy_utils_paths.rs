use clippy_utils::diagnostics::span_lint;
use rustc_ast::ast::{Crate, ItemKind, ModKind};
use rustc_lint::{EarlyContext, EarlyLintPass};
use rustc_lint_defs::declare_tool_lint;
use rustc_session::declare_lint_pass;

declare_tool_lint! {
    /// ### What it does
    /// Checks that [`clippy_utils::paths`] is sorted lexically
    ///
    /// ### Why is this bad?
    /// We like to pretend we're an example of tidy code.
    ///
    /// ### Example
    /// Wrong ordering of the util::paths constants.
    pub clippy::UNSORTED_CLIPPY_UTILS_PATHS,
    Warn,
    "various things that will negatively affect your clippy experience",
    report_in_external_macro: true
}

declare_lint_pass!(UnsortedClippyUtilsPaths => [UNSORTED_CLIPPY_UTILS_PATHS]);

impl EarlyLintPass for UnsortedClippyUtilsPaths {
    fn check_crate(&mut self, cx: &EarlyContext<'_>, krate: &Crate) {
        if let Some(utils) = krate
            .items
            .iter()
            .find(|item| item.kind.ident().is_some_and(|i| i.name.as_str() == "utils"))
            && let ItemKind::Mod(_, _, ModKind::Loaded(ref items, ..)) = utils.kind
            && let Some(paths) = items
                .iter()
                .find(|item| item.kind.ident().is_some_and(|i| i.name.as_str() == "paths"))
            && let ItemKind::Mod(_, _, ModKind::Loaded(ref items, ..)) = paths.kind
        {
            let mut last_name: Option<String> = None;
            for item in items {
                let name = item.kind.ident().expect("const items have idents").to_string();
                if let Some(last_name) = last_name
                    && *last_name > *name
                {
                    span_lint(
                        cx,
                        UNSORTED_CLIPPY_UTILS_PATHS,
                        item.span,
                        "this constant should be before the previous constant due to lexical \
                                         ordering",
                    );
                }
                last_name = Some(name);
            }
        }
    }
}
