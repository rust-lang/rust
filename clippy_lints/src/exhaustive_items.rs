use crate::utils::{indent_of, snippet_opt, span_lint_and_help, span_lint_and_sugg};
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir::{Item, ItemKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::sym;

declare_clippy_lint! {
    /// **What it does:** Warns on any exported `enum`s that are not tagged `#[non_exhaustive]`
    ///
    /// **Why is this bad?** Exhaustive enums are typically fine, but a project which does
    /// not wish to make a stability commitment around exported enums may wish to
    /// disable them by default.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    ///
    /// ```rust
    /// enum Foo {
    ///     Bar,
    ///     Baz
    /// }
    /// ```
    /// Use instead:
    /// ```rust
    /// #[non_exhaustive]
    /// enum Foo {
    ///     Bar,
    ///     Baz
    /// }
    /// ```
    pub EXHAUSTIVE_ENUMS,
    restriction,
    "detects exported enums that have not been marked #[non_exhaustive]"
}

declare_clippy_lint! {
    /// **What it does:** Warns on any exported `structs`s that are not tagged `#[non_exhaustive]`
    ///
    /// **Why is this bad?** Exhaustive structs are typically fine, but a project which does
    /// not wish to make a stability commitment around exported structs may wish to
    /// disable them by default.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    ///
    /// ```rust
    /// struct Foo {
    ///     bar: u8,
    ///     baz: String,
    /// }
    /// ```
    /// Use instead:
    /// ```rust
    /// #[non_exhaustive]
    /// struct Foo {
    ///     bar: u8,
    ///     baz: String,
    /// }
    /// ```
    pub EXHAUSTIVE_STRUCTS,
    restriction,
    "detects exported structs that have not been marked #[non_exhaustive]"
}

declare_lint_pass!(ExhaustiveItems => [EXHAUSTIVE_ENUMS, EXHAUSTIVE_STRUCTS]);

impl LateLintPass<'_> for ExhaustiveItems {
    fn check_item(&mut self, cx: &LateContext<'_>, item: &Item<'_>) {
        if_chain! {
            if let ItemKind::Enum(..) | ItemKind::Struct(..) = item.kind;
            if cx.access_levels.is_exported(item.hir_id);
            if !item.attrs.iter().any(|a| a.has_name(sym::non_exhaustive));
            then {
                let lint = if let ItemKind::Enum(..) = item.kind {
                    EXHAUSTIVE_ENUMS
                } else {
                    EXHAUSTIVE_STRUCTS
                };

                if let Some(snippet) = snippet_opt(cx, item.span) {
                    let indent = " ".repeat(indent_of(cx, item.span).unwrap_or(0));
                    span_lint_and_sugg(
                        cx,
                        lint,
                        item.span,
                        "enums should not be exhaustive",
                        "try adding #[non_exhaustive]",
                        format!("#[non_exhaustive]\n{}{}", indent, snippet),
                        Applicability::MaybeIncorrect,
                    );
                } else {
                    span_lint_and_help(
                        cx,
                        lint,
                        item.span,
                        "enums should not be exhaustive",
                        None,
                        "try adding #[non_exhaustive]",
                    );
                }
            }
        }
    }
}
