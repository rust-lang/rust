use clippy_utils::diagnostics::span_lint_and_then;
use rustc_ast::MetaItemInner;
use rustc_lint::{EarlyContext, EarlyLintPass};
use rustc_session::declare_lint_pass;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of `cfg` that excludes code from `test` builds. (i.e., `#[cfg(not(test))]`)
    ///
    /// ### Why is this bad?
    /// This may give the false impression that a codebase has 100% coverage, yet actually has untested code.
    /// Enabling this also guards against excessive mockery as well, which is an anti-pattern.
    ///
    /// ### Example
    /// ```rust
    /// # fn important_check() {}
    /// #[cfg(not(test))]
    /// important_check(); // I'm not actually tested, but not including me will falsely increase coverage!
    /// ```
    /// Use instead:
    /// ```rust
    /// # fn important_check() {}
    /// important_check();
    /// ```
    #[clippy::version = "1.81.0"]
    pub CFG_NOT_TEST,
    restriction,
    "enforce against excluding code from test builds"
}

declare_lint_pass!(CfgNotTest => [CFG_NOT_TEST]);

impl EarlyLintPass for CfgNotTest {
    fn check_attribute(&mut self, cx: &EarlyContext<'_>, attr: &rustc_ast::Attribute) {
        if attr.has_name(rustc_span::sym::cfg_trace) && contains_not_test(attr.meta_item_list().as_deref(), false) {
            span_lint_and_then(
                cx,
                CFG_NOT_TEST,
                attr.span,
                "code is excluded from test builds",
                |diag| {
                    diag.help("consider not excluding any code from test builds");
                    diag.note_once("this could increase code coverage despite not actually being tested");
                },
            );
        }
    }
}

fn contains_not_test(list: Option<&[MetaItemInner]>, not: bool) -> bool {
    list.is_some_and(|list| {
        list.iter().any(|item| {
            item.ident().is_some_and(|ident| match ident.name {
                rustc_span::sym::not => contains_not_test(item.meta_item_list(), !not),
                rustc_span::sym::test => not,
                _ => contains_not_test(item.meta_item_list(), not),
            })
        })
    })
}
