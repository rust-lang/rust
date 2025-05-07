use rustc_ast::token::Delimiter;
use rustc_errors::Diag;
use rustc_session::parse::ParseSess;
use rustc_span::Span;
use rustc_span::source_map::SourceMap;

use super::UnmatchedDelim;
use crate::errors::MismatchedClosingDelimiter;
use crate::pprust;

#[derive(Default)]
pub(super) struct TokenTreeDiagInfo {
    /// Stack of open delimiters and their spans. Used for error message.
    pub open_delimiters: Vec<(Delimiter, Span)>,
    pub unmatched_delims: Vec<UnmatchedDelim>,

    /// Used only for error recovery when arriving to EOF with mismatched braces.
    pub last_unclosed_found_span: Option<Span>,

    /// Collect empty block spans that might have been auto-inserted by editors.
    pub empty_block_spans: Vec<Span>,

    /// Collect the spans of braces (Open, Close). Used only
    /// for detecting if blocks are empty and only braces.
    pub matching_block_spans: Vec<(Span, Span)>,
}

pub(super) fn same_indentation_level(sm: &SourceMap, open_sp: Span, close_sp: Span) -> bool {
    match (sm.span_to_margin(open_sp), sm.span_to_margin(close_sp)) {
        (Some(open_padding), Some(close_padding)) => open_padding == close_padding,
        _ => false,
    }
}

// When we get a `)` or `]` for `{`, we should emit help message here
// it's more friendly compared to report `unmatched error` in later phase
fn report_missing_open_delim(err: &mut Diag<'_>, unmatched_delims: &[UnmatchedDelim]) -> bool {
    let mut reported_missing_open = false;
    for unmatch_brace in unmatched_delims.iter() {
        if let Some(delim) = unmatch_brace.found_delim
            && matches!(delim, Delimiter::Parenthesis | Delimiter::Bracket)
        {
            let missed_open = match delim {
                Delimiter::Parenthesis => "(",
                Delimiter::Bracket => "[",
                _ => unreachable!(),
            };
            err.span_label(
                unmatch_brace.found_span.shrink_to_lo(),
                format!("missing open `{missed_open}` for this delimiter"),
            );
            reported_missing_open = true;
        }
    }
    reported_missing_open
}

pub(super) fn report_suspicious_mismatch_block(
    err: &mut Diag<'_>,
    diag_info: &TokenTreeDiagInfo,
    sm: &SourceMap,
    delim: Delimiter,
) {
    if report_missing_open_delim(err, &diag_info.unmatched_delims) {
        return;
    }

    let mut matched_spans: Vec<(Span, bool)> = diag_info
        .matching_block_spans
        .iter()
        .map(|&(open, close)| (open.with_hi(close.lo()), same_indentation_level(sm, open, close)))
        .collect();

    // sort by `lo`, so the large block spans in the front
    matched_spans.sort_by_key(|(span, _)| span.lo());

    // We use larger block whose indentation is well to cover those inner mismatched blocks
    // O(N^2) here, but we are on error reporting path, so it is fine
    for i in 0..matched_spans.len() {
        let (block_span, same_ident) = matched_spans[i];
        if same_ident {
            for j in i + 1..matched_spans.len() {
                let (inner_block, inner_same_ident) = matched_spans[j];
                if block_span.contains(inner_block) && !inner_same_ident {
                    matched_spans[j] = (inner_block, true);
                }
            }
        }
    }

    // Find the innermost span candidate for final report
    let candidate_span =
        matched_spans.into_iter().rev().find(|&(_, same_ident)| !same_ident).map(|(span, _)| span);

    if let Some(block_span) = candidate_span {
        err.span_label(block_span.shrink_to_lo(), "this delimiter might not be properly closed...");
        err.span_label(
            block_span.shrink_to_hi(),
            "...as it matches this but it has different indentation",
        );

        // If there is a empty block in the mismatched span, note it
        if delim == Delimiter::Brace {
            for span in diag_info.empty_block_spans.iter() {
                if block_span.contains(*span) {
                    err.span_label(*span, "block is empty, you might have not meant to close it");
                    break;
                }
            }
        }
    } else {
        // If there is no suspicious span, give the last properly closed block may help
        if let Some(parent) = diag_info.matching_block_spans.last()
            && diag_info.open_delimiters.last().is_none()
            && diag_info.empty_block_spans.iter().all(|&sp| sp != parent.0.to(parent.1))
        {
            err.span_label(parent.0, "this opening brace...");
            err.span_label(parent.1, "...matches this closing brace");
        }
    }
}

pub(crate) fn make_unclosed_delims_error(
    unmatched: UnmatchedDelim,
    psess: &ParseSess,
) -> Option<Diag<'_>> {
    // `None` here means an `Eof` was found. We already emit those errors elsewhere, we add them to
    // `unmatched_delims` only for error recovery in the `Parser`.
    let found_delim = unmatched.found_delim?;
    let mut spans = vec![unmatched.found_span];
    if let Some(sp) = unmatched.unclosed_span {
        spans.push(sp);
    };
    let err = psess.dcx().create_err(MismatchedClosingDelimiter {
        spans,
        delimiter: pprust::token_kind_to_string(&found_delim.as_close_token_kind()).to_string(),
        unmatched: unmatched.found_span,
        opening_candidate: unmatched.candidate_span,
        unclosed: unmatched.unclosed_span,
    });
    Some(err)
}
