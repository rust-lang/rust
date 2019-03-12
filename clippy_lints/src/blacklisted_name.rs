use crate::utils::span_lint;
use rustc::hir::*;
use rustc::lint::{LateContext, LateLintPass, LintArray, LintPass};
use rustc::{declare_tool_lint, lint_array};
use rustc_data_structures::fx::FxHashSet;

declare_clippy_lint! {
    /// **What it does:** Checks for usage of blacklisted names for variables, such
    /// as `foo`.
    ///
    /// **Why is this bad?** These names are usually placeholder names and should be
    /// avoided.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust
    /// let foo = 3.14;
    /// ```
    pub BLACKLISTED_NAME,
    style,
    "usage of a blacklisted/placeholder name"
}

#[derive(Clone, Debug)]
pub struct BlackListedName {
    blacklist: FxHashSet<String>,
}

impl BlackListedName {
    pub fn new(blacklist: FxHashSet<String>) -> Self {
        Self { blacklist }
    }
}

impl LintPass for BlackListedName {
    fn get_lints(&self) -> LintArray {
        lint_array!(BLACKLISTED_NAME)
    }
    fn name(&self) -> &'static str {
        "BlacklistedName"
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for BlackListedName {
    fn check_pat(&mut self, cx: &LateContext<'a, 'tcx>, pat: &'tcx Pat) {
        if let PatKind::Binding(.., ident, _) = pat.node {
            if self.blacklist.contains(&ident.name.to_string()) {
                span_lint(
                    cx,
                    BLACKLISTED_NAME,
                    ident.span,
                    &format!("use of a blacklisted/placeholder name `{}`", ident.name),
                );
            }
        }
    }
}
