use clippy_utils::diagnostics::span_lint;
use clippy_utils::is_test_module_or_function;
use rustc_data_structures::fx::FxHashSet;
use rustc_hir::{Item, Pat, PatKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_tool_lint, impl_lint_pass};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of disallowed names for variables, such
    /// as `foo`.
    ///
    /// ### Why is this bad?
    /// These names are usually placeholder names and should be
    /// avoided.
    ///
    /// ### Example
    /// ```rust
    /// let foo = 3.14;
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub DISALLOWED_NAMES,
    style,
    "usage of a disallowed/placeholder name"
}

#[derive(Clone, Debug)]
pub struct DisallowedNames {
    disallow: FxHashSet<String>,
    test_modules_deep: u32,
}

impl DisallowedNames {
    pub fn new(disallow: FxHashSet<String>) -> Self {
        Self {
            disallow,
            test_modules_deep: 0,
        }
    }

    fn in_test_module(&self) -> bool {
        self.test_modules_deep != 0
    }
}

impl_lint_pass!(DisallowedNames => [DISALLOWED_NAMES]);

impl<'tcx> LateLintPass<'tcx> for DisallowedNames {
    fn check_item(&mut self, cx: &LateContext<'_>, item: &Item<'_>) {
        if is_test_module_or_function(cx.tcx, item) {
            self.test_modules_deep = self.test_modules_deep.saturating_add(1);
        }
    }

    fn check_pat(&mut self, cx: &LateContext<'tcx>, pat: &'tcx Pat<'_>) {
        // Check whether we are under the `test` attribute.
        if self.in_test_module() {
            return;
        }

        if let PatKind::Binding(.., ident, _) = pat.kind {
            if self.disallow.contains(&ident.name.to_string()) {
                span_lint(
                    cx,
                    DISALLOWED_NAMES,
                    ident.span,
                    &format!("use of a disallowed/placeholder name `{}`", ident.name),
                );
            }
        }
    }

    fn check_item_post(&mut self, cx: &LateContext<'_>, item: &Item<'_>) {
        if is_test_module_or_function(cx.tcx, item) {
            self.test_modules_deep = self.test_modules_deep.saturating_sub(1);
        }
    }
}
