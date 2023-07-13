use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet_opt;
use rustc_ast::ast::{Item, VisibilityKind};
use rustc_errors::Applicability;
use rustc_lint::{EarlyContext, EarlyLintPass, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::symbol::kw;
use rustc_span::Span;

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
    /// ### Why is this bad?
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
    /// ### Why is this bad?
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
        if !in_external_macro(cx.sess(), item.span)
            && let VisibilityKind::Restricted { path, shorthand, .. } = &item.vis.kind
        {
            if **path == kw::SelfLower && let Some(false) = is_from_proc_macro(cx, item.vis.span) {
                span_lint_and_sugg(
                    cx,
                    NEEDLESS_PUB_SELF,
                    item.vis.span,
                    &format!("unnecessary `pub({}self)`", if *shorthand { "" } else { "in " }),
                    "remove it",
                    String::new(),
                    Applicability::MachineApplicable,
                );
            }

            if (**path == kw::Super || **path == kw::SelfLower || **path == kw::Crate)
                && !*shorthand
                && let [.., last] = &*path.segments
                && let Some(false) = is_from_proc_macro(cx, item.vis.span)
            {
                span_lint_and_sugg(
                    cx,
                    PUB_WITHOUT_SHORTHAND,
                    item.vis.span,
                    "usage of `pub` with `in`",
                    "remove it",
                    format!("pub({})", last.ident),
                    Applicability::MachineApplicable,
                );
            }

            if *shorthand
                && let [.., last] = &*path.segments
                && let Some(false) = is_from_proc_macro(cx, item.vis.span)
            {
                span_lint_and_sugg(
                    cx,
                    PUB_WITH_SHORTHAND,
                    item.vis.span,
                    "usage of `pub` without `in`",
                    "add it",
                    format!("pub(in {})", last.ident),
                    Applicability::MachineApplicable,
                );
            }
        }
    }
}

fn is_from_proc_macro(cx: &EarlyContext<'_>, span: Span) -> Option<bool> {
    snippet_opt(cx, span).map(|s| !s.starts_with("pub"))
}
