use clippy_utils::diagnostics::span_lint_and_then;
use rustc_ast::{Item, ItemKind, UseTreeKind};
use rustc_errors::Applicability;
use rustc_lint::{EarlyContext, EarlyLintPass};
use rustc_session::declare_lint_pass;
use rustc_span::symbol::kw;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for imports ending in `::{self}`.
    ///
    /// ### Why restrict this?
    /// In most cases, this can be written much more cleanly by omitting `::{self}`.
    ///
    /// ### Known problems
    /// Removing `::{self}` will cause any non-module items at the same path to also be imported.
    /// This might cause a naming conflict (https://github.com/rust-lang/rustfmt/issues/3568). This lint makes no attempt
    /// to detect this scenario and that is why it is a restriction lint.
    ///
    /// ### Example
    /// ```no_run
    /// use std::io::{self};
    /// ```
    /// Use instead:
    /// ```no_run
    /// use std::io;
    /// ```
    #[clippy::version = "1.53.0"]
    pub UNNECESSARY_SELF_IMPORTS,
    restriction,
    "imports ending in `::{self}`, which can be omitted"
}

declare_lint_pass!(UnnecessarySelfImports => [UNNECESSARY_SELF_IMPORTS]);

impl EarlyLintPass for UnnecessarySelfImports {
    fn check_item(&mut self, cx: &EarlyContext<'_>, item: &Item) {
        if let ItemKind::Use(use_tree) = &item.kind
            && let UseTreeKind::Nested { items, .. } = &use_tree.kind
            && let [(self_tree, _)] = &**items
            && let [self_seg] = &*self_tree.prefix.segments
            && self_seg.ident.name == kw::SelfLower
            && let Some(last_segment) = use_tree.prefix.segments.last()
        {
            span_lint_and_then(
                cx,
                UNNECESSARY_SELF_IMPORTS,
                item.span,
                "import ending with `::{self}`",
                |diag| {
                    diag.span_suggestion(
                        last_segment.span().with_hi(item.span.hi()),
                        "consider omitting `::{self}`",
                        format!(
                            "{}{};",
                            last_segment.ident,
                            if let UseTreeKind::Simple(Some(alias)) = self_tree.kind {
                                format!(" as {alias}")
                            } else {
                                String::new()
                            },
                        ),
                        Applicability::MaybeIncorrect,
                    );
                    diag.note("this will slightly change semantics; any non-module items at the same path will also be imported");
                },
            );
        }
    }
}
