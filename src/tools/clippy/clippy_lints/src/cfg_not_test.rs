use clippy_utils::diagnostics::span_lint_and_then;
use rustc_lint::{EarlyContext, EarlyLintPass};
use rustc_session::declare_lint_pass;
use rustc_ast::AttrItemKind;
use rustc_ast::EarlyParsedAttribute;
use rustc_span::sym;
use rustc_ast::attr::data_structures::CfgEntry;

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
        if attr.has_name(sym::cfg_trace) {
            let AttrItemKind::Parsed(EarlyParsedAttribute::CfgTrace(cfg)) = &attr.get_normal_item().args else {
                unreachable!()
            };

            if contains_not_test(&cfg, false) {
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
}

fn contains_not_test(cfg: &CfgEntry, not: bool) -> bool {
    match cfg {
        CfgEntry::All(subs, _) | CfgEntry::Any(subs, _) => subs.iter().any(|item| {
            contains_not_test(item, not)
        }),
        CfgEntry::Not(sub, _) => contains_not_test(sub, !not),
        CfgEntry::NameValue { name: sym::test, .. } => not,
        _ => false
    }
}
