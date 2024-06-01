use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::source::snippet_opt;
use rustc_errors::Applicability;
use rustc_lint::LateContext;
use rustc_span::Span;

use super::DOC_COMMENT_DOUBLE_SPACE_LINEBREAK;

pub fn check(cx: &LateContext<'_>, collected_breaks: &[Span]) {
    let replacements: Vec<_> = collect_doc_replacements(cx, collected_breaks);

    if let Some((&(lo_span, _), &(hi_span, _))) = replacements.first().zip(replacements.last()) {
        span_lint_and_then(
            cx,
            DOC_COMMENT_DOUBLE_SPACE_LINEBREAK,
            lo_span.to(hi_span),
            "doc comment uses two spaces for a hard line break",
            |diag| {
                diag.multipart_suggestion(
                    "replace this double space with a backslash",
                    replacements,
                    Applicability::MachineApplicable,
                );
            },
        );
    }
}

fn collect_doc_replacements(cx: &LateContext<'_>, spans: &[Span]) -> Vec<(Span, String)> {
    spans
        .iter()
        .map(|span| {
            // we already made sure the snippet exists when collecting spans
            let s = snippet_opt(cx, *span).expect("snippet was already validated to exist");
            let after_newline = s.trim_start_matches(' ');

            let new_comment = format!("\\{after_newline}");
            (*span, new_comment)
        })
        .collect()
}
