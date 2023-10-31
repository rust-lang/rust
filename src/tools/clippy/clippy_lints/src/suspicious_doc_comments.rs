use clippy_utils::diagnostics::{multispan_sugg_with_applicability, span_lint_and_then};
use if_chain::if_chain;
use rustc_ast::token::CommentKind;
use rustc_ast::{AttrKind, AttrStyle, Attribute, Item};
use rustc_errors::Applicability;
use rustc_lint::{EarlyContext, EarlyLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::Span;

declare_clippy_lint! {
    /// ### What it does
    /// Detects the use of outer doc comments (`///`, `/**`) followed by a bang (`!`): `///!`
    ///
    /// ### Why is this bad?
    /// Triple-slash comments (known as "outer doc comments") apply to items that follow it.
    /// An outer doc comment followed by a bang (i.e. `///!`) has no specific meaning.
    ///
    /// The user most likely meant to write an inner doc comment (`//!`, `/*!`), which
    /// applies to the parent item (i.e. the item that the comment is contained in,
    /// usually a module or crate).
    ///
    /// ### Known problems
    /// Inner doc comments can only appear before items, so there are certain cases where the suggestion
    /// made by this lint is not valid code. For example:
    /// ```rs
    /// fn foo() {}
    /// ///!
    /// fn bar() {}
    /// ```
    /// This lint detects the doc comment and suggests changing it to `//!`, but an inner doc comment
    /// is not valid at that position.
    ///
    /// ### Example
    /// In this example, the doc comment is attached to the *function*, rather than the *module*.
    /// ```rust
    /// pub mod util {
    ///     ///! This module contains utility functions.
    ///
    ///     pub fn dummy() {}
    /// }
    /// ```
    ///
    /// Use instead:
    /// ```rust
    /// pub mod util {
    ///     //! This module contains utility functions.
    ///
    ///     pub fn dummy() {}
    /// }
    /// ```
    #[clippy::version = "1.70.0"]
    pub SUSPICIOUS_DOC_COMMENTS,
    suspicious,
    "suspicious usage of (outer) doc comments"
}
declare_lint_pass!(SuspiciousDocComments => [SUSPICIOUS_DOC_COMMENTS]);

const WARNING: &str = "this is an outer doc comment and does not apply to the parent module or crate";
const HELP: &str = "use an inner doc comment to document the parent module or crate";

impl EarlyLintPass for SuspiciousDocComments {
    fn check_item(&mut self, cx: &EarlyContext<'_>, item: &Item) {
        let replacements = collect_doc_comment_replacements(&item.attrs);

        if let Some(((lo_span, _), (hi_span, _))) = replacements.first().zip(replacements.last()) {
            let span = lo_span.to(*hi_span);

            span_lint_and_then(cx, SUSPICIOUS_DOC_COMMENTS, span, WARNING, |diag| {
                multispan_sugg_with_applicability(diag, HELP, Applicability::MaybeIncorrect, replacements);
            });
        }
    }
}

fn collect_doc_comment_replacements(attrs: &[Attribute]) -> Vec<(Span, String)> {
    attrs
        .iter()
        .filter_map(|attr| {
            if_chain! {
                if let AttrKind::DocComment(com_kind, sym) = attr.kind;
                if let AttrStyle::Outer = attr.style;
                if let Some(com) = sym.as_str().strip_prefix('!');
                then {
                    let sugg = match com_kind {
                        CommentKind::Line => format!("//!{com}"),
                        CommentKind::Block => format!("/*!{com}*/")
                    };
                    Some((attr.span, sugg))
                } else {
                    None
                }
            }
        })
        .collect()
}
