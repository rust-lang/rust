use crate::utils::{snippet_opt, span_lint_and_help, span_lint_and_sugg};
use if_chain::if_chain;
use rustc_hir::{Item, ItemKind};
use rustc_errors::Applicability;
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::sym;

declare_clippy_lint! {
    /// **What it does:** Warns on any `enum`s that are not tagged `#[non_exhaustive]`
    ///
    /// **Why is this bad?** Exhaustive enums are typically fine, but a project which does
    /// not wish to make a stability commitment around enums may wish to disable them by default.
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
    /// }    /// ```
    pub EXHAUSTIVE_ENUMS,
    restriction,
    "default lint description"
}

declare_lint_pass!(ExhaustiveEnums => [EXHAUSTIVE_ENUMS]);

impl LateLintPass<'_> for ExhaustiveEnums {
    fn check_item(&mut self, cx: &LateContext<'_>, item: &Item<'_>) {
        if_chain! {
            if let ItemKind::Enum(..) = item.kind;
            if !item.attrs.iter().any(|a| a.has_name(sym::non_exhaustive));
            then {
                if let Some(snippet) = snippet_opt(cx, item.span) {
                    span_lint_and_sugg(
                        cx,
                        EXHAUSTIVE_ENUMS,
                        item.span,
                        "enums should not be exhaustive",
                        "try adding #[non_exhaustive]",
                        format!("#[non_exhaustive]\n{}", snippet),
                        Applicability::MaybeIncorrect,
                    );
                } else {
                    span_lint_and_help(
                        cx,
                        EXHAUSTIVE_ENUMS,
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
