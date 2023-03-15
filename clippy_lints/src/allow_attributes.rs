use ast::AttrStyle;
use clippy_utils::diagnostics::span_lint_and_sugg;
use rustc_ast as ast;
use rustc_errors::Applicability;
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// Detects uses of the `#[allow]` attribute and suggests replacing it with
    /// the `#[expect]` (See [RFC 2383](https://rust-lang.github.io/rfcs/2383-lint-reasons.html))
    ///
    /// The expect attribute is still unstable and requires the `lint_reasons`
    /// on nightly. It can be enabled by adding `#![feature(lint_reasons)]` to
    /// the crate root.
    ///
    /// This lint only warns outer attributes (`#[allow]`), as inner attributes
    /// (`#![allow]`) are usually used to enable or disable lints on a global scale.
    ///
    /// ### Why is this bad?
    ///
    /// `#[expect]` attributes suppress the lint emission, but emit a warning, if
    /// the expectation is unfulfilled. This can be useful to be notified when the
    /// lint is no longer triggered.
    ///
    /// ### Example
    /// ```rust,ignore
    /// #[allow(unused_mut)]
    /// fn foo() -> usize {
    ///    let mut a = Vec::new();
    ///    a.len()
    /// }
    /// ```
    /// Use instead:
    /// ```rust,ignore
    /// #![feature(lint_reasons)]
    /// #[expect(unused_mut)]
    /// fn foo() -> usize {
    ///     let mut a = Vec::new();
    ///     a.len()
    /// }
    /// ```
    #[clippy::version = "1.69.0"]
    pub ALLOW_ATTRIBUTES,
    restriction,
    "`#[allow]` will not trigger if a warning isn't found. `#[expect]` triggers if there are no warnings."
}

declare_lint_pass!(AllowAttribute => [ALLOW_ATTRIBUTES]);

impl LateLintPass<'_> for AllowAttribute {
    // Separate each crate's features.
    fn check_attribute(&mut self, cx: &LateContext<'_>, attr: &ast::Attribute) {
        if_chain! {
            if cx.tcx.features().lint_reasons;
            if let AttrStyle::Outer = attr.style;
            if let Some(ident) = attr.ident();
            if ident.name == rustc_span::symbol::sym::allow;
            then {
                span_lint_and_sugg(
                    cx,
                    ALLOW_ATTRIBUTES,
                    ident.span,
                    "#[allow] attribute found",
                    "replace it with",
                    "expect".into(),
                    Applicability::MachineApplicable,
                );
            }
        }
    }
}
