use clippy_utils::diagnostics::span_lint_and_help;
use rustc_ast::ast::{Item, ItemKind, VisibilityKind};
use rustc_lint::{EarlyContext, EarlyLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// ### What it does
    ///
    /// Restricts the usage of `pub use ...`
    ///
    /// ### Why is this bad?
    ///
    /// `pub use` is usually fine, but a project may wish to limit `pub use` instances to prevent
    /// unintentional exports or to encourage placing exported items directly in public modules
    ///
    /// ### Example
    /// ```rust
    /// pub mod outer {
    ///     mod inner {
    ///         pub struct Test {}
    ///     }
    ///     pub use inner::Test;
    /// }
    ///
    /// use outer::Test;
    /// ```
    /// Use instead:
    /// ```rust
    /// pub mod outer {
    ///     pub struct Test {}
    /// }
    ///
    /// use outer::Test;
    /// ```
    #[clippy::version = "1.62.0"]
    pub PUB_USE,
    restriction,
    "restricts the usage of `pub use`"
}
declare_lint_pass!(PubUse => [PUB_USE]);

impl EarlyLintPass for PubUse {
    fn check_item(&mut self, cx: &EarlyContext<'_>, item: &Item) {
        if let ItemKind::Use(_) = item.kind &&
            let VisibilityKind::Public = item.vis.kind {
                span_lint_and_help(
                    cx,
                    PUB_USE,
                    item.span,
                    "using `pub use`",
                    None,
                    "move the exported item to a public module instead",
                );
            }
    }
}
