use ast::{AttrStyle, MetaItemKind};
use clippy_utils::{diagnostics::span_lint_and_sugg, source::snippet};
use rustc_ast as ast;
use rustc_errors::Applicability;
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::{symbol::Ident, BytePos};

declare_clippy_lint! {
    /// ### What it does
    /// Detects uses of the `#[allow]` attribute and suggests to replace it with the new `#[expect]` attribute implemented by `#![feature(lint_reasons)]` ([RFC 2383](https://rust-lang.github.io/rfcs/2383-lint-reasons.html))
    /// ### Why is this bad?
    /// Using `#[allow]` isn't bad, but `#[expect]` may be preferred as it lints if the code **doesn't** produce a warning.
    /// ### Example
    /// ```rust
    /// #[allow(unused_mut)]
    /// fn foo() -> usize {
    ///    let mut a = Vec::new();
    ///    a.len()
    ///}
    /// ```
    /// Use instead:
    /// ```rust
    /// # #![feature(lint_reasons)]
    /// #[expect(unused_mut)]
    /// fn foo() -> usize {
    ///     let mut a = Vec::new();
    ///     a.len()
    /// }
    /// ```
    #[clippy::version = "1.69.0"]
    pub ALLOW_ATTRIBUTE,
    restriction,
    "`#[allow]` will not trigger if a warning isn't found. `#[expect]` triggers if there are no warnings."
}

pub struct AllowAttribute {
    pub lint_reasons_active: bool,
}

impl_lint_pass!(AllowAttribute => [ALLOW_ATTRIBUTE]);

impl LateLintPass<'_> for AllowAttribute {
    // Separate each crate's features.
    fn check_crate_post(&mut self, _: &LateContext<'_>) {
        self.lint_reasons_active = false;
    }
    fn check_attribute(&mut self, cx: &LateContext<'_>, attr: &ast::Attribute) {
        // Check inner attributes

        if_chain! {
            if let AttrStyle::Inner = attr.style;
            if attr.ident()
            .unwrap_or(Ident::with_dummy_span(sym!(empty))) // Will not trigger if doesn't have an ident.
            .name == sym!(feature);
            if let ast::AttrKind::Normal(normal) = &attr.kind;
            if let Some(MetaItemKind::List(list)) = normal.item.meta_kind();
            if list[0].ident().unwrap().name == sym!(lint_reasons);
            then {
                self.lint_reasons_active = true;
            }
        }

        // Check outer attributes

        if_chain! {
            if let AttrStyle::Outer = attr.style;
            if attr.ident()
            .unwrap_or(Ident::with_dummy_span(sym!(empty))) // Will not trigger if doesn't have an ident.
            .name == sym!(allow);
            if self.lint_reasons_active;
            then {
                span_lint_and_sugg(
                    cx,
                    ALLOW_ATTRIBUTE,
                    attr.span,
                    "#[allow] attribute found",
                    "replace it with",
                    format!("#[expect{})]", snippet(
                        cx,
                        attr.ident().unwrap().span
                        .with_lo(
                            attr.ident().unwrap().span.hi() + BytePos(2) // Cut [(
                        )
                        .with_hi(
                            attr.meta().unwrap().span.hi() - BytePos(2) // Cut )]
                        )
                        , "...")), Applicability::MachineApplicable);
            }
        }
    }
}
