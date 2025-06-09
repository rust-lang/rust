use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::source::indent_of;
use rustc_errors::Applicability;
use rustc_hir::{Item, ItemKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::declare_lint_pass;
use rustc_span::sym;

declare_clippy_lint! {
    /// ### What it does
    /// Warns on any exported `enum`s that are not tagged `#[non_exhaustive]`
    ///
    /// ### Why restrict this?
    /// Making an `enum` exhaustive is a stability commitment: adding a variant is a breaking change.
    /// A project may wish to ensure that there are no exhaustive enums or that every exhaustive
    /// `enum` is explicitly `#[allow]`ed.
    ///
    /// ### Example
    /// ```no_run
    /// enum Foo {
    ///     Bar,
    ///     Baz
    /// }
    /// ```
    /// Use instead:
    /// ```no_run
    /// #[non_exhaustive]
    /// enum Foo {
    ///     Bar,
    ///     Baz
    /// }
    /// ```
    #[clippy::version = "1.51.0"]
    pub EXHAUSTIVE_ENUMS,
    restriction,
    "detects exported enums that have not been marked #[non_exhaustive]"
}

declare_clippy_lint! {
    /// ### What it does
    /// Warns on any exported `struct`s that are not tagged `#[non_exhaustive]`
    ///
    /// ### Why restrict this?
    /// Making a `struct` exhaustive is a stability commitment: adding a field is a breaking change.
    /// A project may wish to ensure that there are no exhaustive structs or that every exhaustive
    /// `struct` is explicitly `#[allow]`ed.
    ///
    /// ### Example
    /// ```no_run
    /// struct Foo {
    ///     bar: u8,
    ///     baz: String,
    /// }
    /// ```
    /// Use instead:
    /// ```no_run
    /// #[non_exhaustive]
    /// struct Foo {
    ///     bar: u8,
    ///     baz: String,
    /// }
    /// ```
    #[clippy::version = "1.51.0"]
    pub EXHAUSTIVE_STRUCTS,
    restriction,
    "detects exported structs that have not been marked #[non_exhaustive]"
}

declare_lint_pass!(ExhaustiveItems => [EXHAUSTIVE_ENUMS, EXHAUSTIVE_STRUCTS]);

impl LateLintPass<'_> for ExhaustiveItems {
    fn check_item(&mut self, cx: &LateContext<'_>, item: &Item<'_>) {
        let (lint, msg, fields) = match item.kind {
            ItemKind::Enum(..) => (
                EXHAUSTIVE_ENUMS,
                "exported enums should not be exhaustive",
                [].as_slice(),
            ),
            ItemKind::Struct(_, _, v) => (
                EXHAUSTIVE_STRUCTS,
                "exported structs should not be exhaustive",
                v.fields(),
            ),
            _ => return,
        };
        if cx.effective_visibilities.is_exported(item.owner_id.def_id)
            && let attrs = cx.tcx.hir_attrs(item.hir_id())
            && !attrs.iter().any(|a| a.has_name(sym::non_exhaustive))
            && fields.iter().all(|f| cx.tcx.visibility(f.def_id).is_public())
        {
            span_lint_and_then(cx, lint, item.span, msg, |diag| {
                let suggestion_span = item.span.shrink_to_lo();
                let indent = " ".repeat(indent_of(cx, item.span).unwrap_or(0));
                let sugg = format!("#[non_exhaustive]\n{indent}");
                diag.span_suggestion_verbose(
                    suggestion_span,
                    "try adding #[non_exhaustive]",
                    sugg,
                    Applicability::MaybeIncorrect,
                );
            });
        }
    }
}
