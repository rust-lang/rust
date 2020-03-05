use crate::utils::{snippet, span_lint_and_sugg};
use if_chain::if_chain;
use rustc_ast::ast;
use rustc_errors::Applicability;
use rustc_lint::{EarlyContext, EarlyLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::edition::Edition;

declare_clippy_lint! {
    /// **What it does:** Checks for `#[macro_use] use...`.
    ///
    /// **Why is this bad?** Since the Rust 2018 edition you can import
    /// macro's directly, this is considered idiomatic.
    ///
    /// **Known problems:** This lint does not generate an auto-applicable suggestion.
    ///
    /// **Example:**
    /// ```rust
    /// #[macro_use]
    /// use lazy_static;
    /// ```
    pub MACRO_USE_IMPORTS,
    pedantic,
    "#[macro_use] is no longer needed"
}

declare_lint_pass!(MacroUseImports => [MACRO_USE_IMPORTS]);

impl EarlyLintPass for MacroUseImports {
    fn check_item(&mut self, ecx: &EarlyContext<'_>, item: &ast::Item) {
        if_chain! {
            if ecx.sess.opts.edition == Edition::Edition2018;
            if let ast::ItemKind::Use(use_tree) = &item.kind;
            if let Some(mac_attr) = item
                .attrs
                .iter()
                .find(|attr| attr.ident().map(|s| s.to_string()) == Some("macro_use".to_string()));
            then {
                let msg = "`macro_use` attributes are no longer needed in the Rust 2018 edition";
                let help = format!("use {}::<macro name>", snippet(ecx, use_tree.span, "_"));
                span_lint_and_sugg(
                    ecx,
                    MACRO_USE_IMPORTS,
                    mac_attr.span,
                    msg,
                    "remove the attribute and import the macro directly, try",
                    help,
                    Applicability::HasPlaceholders,
                );
            }
        }
    }
}
