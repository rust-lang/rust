use clippy_utils::diagnostics::span_lint_and_then;
use rustc_ast::AttrStyle;
use rustc_ast::token::CommentKind;
use rustc_errors::Applicability;
use rustc_hir::Attribute;
use rustc_hir::attrs::AttributeKind;
use rustc_lint::LateContext;
use rustc_span::Span;

use super::SUSPICIOUS_DOC_COMMENTS;

pub fn check(cx: &LateContext<'_>, attrs: &[Attribute]) -> bool {
    let replacements: Vec<_> = collect_doc_replacements(attrs);

    if let Some((&(lo_span, _), &(hi_span, _))) = replacements.first().zip(replacements.last()) {
        span_lint_and_then(
            cx,
            SUSPICIOUS_DOC_COMMENTS,
            lo_span.to(hi_span),
            "this is an outer doc comment and does not apply to the parent module or crate",
            |diag| {
                diag.multipart_suggestion(
                    "use an inner doc comment to document the parent module or crate",
                    replacements,
                    Applicability::MaybeIncorrect,
                );
            },
        );

        true
    } else {
        false
    }
}

fn collect_doc_replacements(attrs: &[Attribute]) -> Vec<(Span, String)> {
    attrs
        .iter()
        .filter_map(|attr| {
            if let Attribute::Parsed(AttributeKind::DocComment {
                style: AttrStyle::Outer,
                kind,
                comment,
                ..
            }) = attr
                && let Some(com) = comment.as_str().strip_prefix('!')
            {
                let sugg = match kind {
                    CommentKind::Line => format!("//!{com}"),
                    CommentKind::Block => format!("/*!{com}*/"),
                };
                Some((attr.span(), sugg))
            } else {
                None
            }
        })
        .collect()
}
