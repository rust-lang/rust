use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::source::SpanRangeExt;
use rustc_ast::ast::{Item, VisibilityKind};
use rustc_errors::Applicability;
use rustc_lint::{EarlyContext, EarlyLintPass, LintContext};
use rustc_session::declare_lint_pass;
use rustc_span::Span;
use rustc_span::symbol::kw;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of `pub(self)` and `pub(in self)`.
    ///
    /// ### Why is this bad?
    /// It's unnecessary, omitting the `pub` entirely will give the same results.
    ///
    /// ### Example
    /// ```rust,ignore
    /// pub(self) type OptBox<T> = Option<Box<T>>;
    /// ```
    /// Use instead:
    /// ```rust,ignore
    /// type OptBox<T> = Option<Box<T>>;
    /// ```
    #[clippy::version = "1.72.0"]
    pub NEEDLESS_PUB_SELF,
    style,
    "checks for usage of `pub(self)` and `pub(in self)`."
}
declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of `pub(<loc>)` with `in`.
    ///
    /// ### Why restrict this?
    /// Consistency. Use it or don't, just be consistent about it.
    ///
    /// Also see the `pub_without_shorthand` lint for an alternative.
    ///
    /// ### Example
    /// ```rust,ignore
    /// pub(super) type OptBox<T> = Option<Box<T>>;
    /// ```
    /// Use instead:
    /// ```rust,ignore
    /// pub(in super) type OptBox<T> = Option<Box<T>>;
    /// ```
    #[clippy::version = "1.72.0"]
    pub PUB_WITH_SHORTHAND,
    restriction,
    "disallows usage of `pub(<loc>)`, without `in`"
}
declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of `pub(<loc>)` without `in`.
    ///
    /// Note: As you cannot write a module's path in `pub(<loc>)`, this will only trigger on
    /// `pub(super)` and the like.
    ///
    /// ### Why restrict this?
    /// Consistency. Use it or don't, just be consistent about it.
    ///
    /// Also see the `pub_with_shorthand` lint for an alternative.
    ///
    /// ### Example
    /// ```rust,ignore
    /// pub(in super) type OptBox<T> = Option<Box<T>>;
    /// ```
    /// Use instead:
    /// ```rust,ignore
    /// pub(super) type OptBox<T> = Option<Box<T>>;
    /// ```
    #[clippy::version = "1.72.0"]
    pub PUB_WITHOUT_SHORTHAND,
    restriction,
    "disallows usage of `pub(in <loc>)` with `in`"
}
declare_lint_pass!(Visibility => [NEEDLESS_PUB_SELF, PUB_WITH_SHORTHAND, PUB_WITHOUT_SHORTHAND]);

impl EarlyLintPass for Visibility {
    fn check_item(&mut self, cx: &EarlyContext<'_>, item: &Item) {
        if !item.span.in_external_macro(cx.sess().source_map())
            && let VisibilityKind::Restricted { path, shorthand, .. } = &item.vis.kind
        {
            if **path == kw::SelfLower && !is_from_proc_macro(cx, item.vis.span) {
                span_lint_and_then(
                    cx,
                    NEEDLESS_PUB_SELF,
                    item.vis.span,
                    format!("unnecessary `pub({}self)`", if *shorthand { "" } else { "in " }),
                    |diag| {
                        diag.span_suggestion_hidden(
                            item.vis.span,
                            "remove it",
                            String::new(),
                            Applicability::MachineApplicable,
                        );
                    },
                );
            }

            if (**path == kw::Super || **path == kw::SelfLower || **path == kw::Crate)
                && !*shorthand
                && let [.., last] = &*path.segments
                && !is_from_proc_macro(cx, item.vis.span)
            {
                #[expect(clippy::collapsible_span_lint_calls, reason = "rust-clippy#7797")]
                span_lint_and_then(
                    cx,
                    PUB_WITHOUT_SHORTHAND,
                    item.vis.span,
                    "usage of `pub` with `in`",
                    |diag| {
                        diag.span_suggestion(
                            item.vis.span,
                            "remove it",
                            format!("pub({})", last.ident),
                            Applicability::MachineApplicable,
                        );
                    },
                );
            }

            if *shorthand
                && let [.., last] = &*path.segments
                && !is_from_proc_macro(cx, item.vis.span)
            {
                #[expect(clippy::collapsible_span_lint_calls, reason = "rust-clippy#7797")]
                span_lint_and_then(
                    cx,
                    PUB_WITH_SHORTHAND,
                    item.vis.span,
                    "usage of `pub` without `in`",
                    |diag| {
                        diag.span_suggestion(
                            item.vis.span,
                            "add it",
                            format!("pub(in {})", last.ident),
                            Applicability::MachineApplicable,
                        );
                    },
                );
            }
        }
    }
}

fn is_from_proc_macro(cx: &EarlyContext<'_>, span: Span) -> bool {
    !span.check_source_text(cx, |src| src.starts_with("pub"))
}
