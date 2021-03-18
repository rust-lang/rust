use crate::utils::{in_macro, span_lint_and_sugg};
use if_chain::if_chain;
use rustc_ast::{Item, ItemKind, UseTreeKind};
use rustc_errors::Applicability;
use rustc_lint::{EarlyContext, EarlyLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::edition::Edition;

declare_clippy_lint! {
    /// **What it does:** Checking for imports with single component use path.
    ///
    /// **Why is this bad?** Import with single component use path such as `use cratename;`
    /// is not necessary, and thus should be removed.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    ///
    /// ```rust,ignore
    /// use regex;
    ///
    /// fn main() {
    ///     regex::Regex::new(r"^\d{4}-\d{2}-\d{2}$").unwrap();
    /// }
    /// ```
    /// Better as
    /// ```rust,ignore
    /// fn main() {
    ///     regex::Regex::new(r"^\d{4}-\d{2}-\d{2}$").unwrap();
    /// }
    /// ```
    pub SINGLE_COMPONENT_PATH_IMPORTS,
    style,
    "imports with single component path are redundant"
}

declare_lint_pass!(SingleComponentPathImports => [SINGLE_COMPONENT_PATH_IMPORTS]);

impl EarlyLintPass for SingleComponentPathImports {
    fn check_item(&mut self, cx: &EarlyContext<'_>, item: &Item) {
        if_chain! {
            if !in_macro(item.span);
            if cx.sess.opts.edition >= Edition::Edition2018;
            if !item.vis.kind.is_pub();
            if let ItemKind::Use(use_tree) = &item.kind;
            if let segments = &use_tree.prefix.segments;
            if segments.len() == 1;
            if let UseTreeKind::Simple(None, _, _) = use_tree.kind;
            then {
                span_lint_and_sugg(
                    cx,
                    SINGLE_COMPONENT_PATH_IMPORTS,
                    item.span,
                    "this import is redundant",
                    "remove it entirely",
                    String::new(),
                    Applicability::MachineApplicable
                );
            }
        }
    }
}
