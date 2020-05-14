use rustc_lint::{LateLintPass, LateContext};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_hir::*;

declare_clippy_lint! {
    /// **What it does:**
    ///
    /// **Why is this bad?**
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    ///
    /// ```rust
    /// // example code where clippy issues a warning
    /// ```
    /// Use instead:
    /// ```rust
    /// // example code which does not raise clippy warning
    /// ```
    pub SORT_BY_KEY_REVERSE,
    complexity,
    "default lint description"
}

declare_lint_pass!(SortByKeyReverse => [SORT_BY_KEY_REVERSE]);

impl LateLintPass<'_, '_> for SortByKeyReverse {}
