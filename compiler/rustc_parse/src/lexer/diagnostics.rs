use super::UnmatchedDelim;
use rustc_ast::token::Delimiter;
use rustc_errors::Diagnostic;
use rustc_span::source_map::SourceMap;
use rustc_span::Span;

#[derive(Default)]
pub struct TokenTreeDiagInfo {
    /// Stack of open delimiters and their spans. Used for error message.
    pub open_braces: Vec<(Delimiter, Span)>,
    pub unmatched_delims: Vec<UnmatchedDelim>,

    /// Used only for error recovery when arriving to EOF with mismatched braces.
    pub last_unclosed_found_span: Option<Span>,

    /// Collect empty block spans that might have been auto-inserted by editors.
    pub empty_block_spans: Vec<Span>,

    /// Collect the spans of braces (Open, Close). Used only
    /// for detecting if blocks are empty and only braces.
    pub matching_block_spans: Vec<(Span, Span)>,
}

pub fn same_identation_level(sm: &SourceMap, open_sp: Span, close_sp: Span) -> bool {
    match (sm.span_to_margin(open_sp), sm.span_to_margin(close_sp)) {
        (Some(open_padding), Some(close_padding)) => open_padding == close_padding,
        _ => false,
    }
}

// When we get a `)` or `]` for `{`, we should emit help message here
// it's more friendly compared to report `unmatched error` in later phase
pub fn report_missing_open_delim(
    err: &mut Diagnostic,
    unmatched_delims: &[UnmatchedDelim],
) -> bool {
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
                format!("missing open `{}` for this delimiter", missed_open),
            );
            reported_missing_open = true;
        }
    }
    reported_missing_open
}

pub fn report_suspicious_mismatch_block(
    err: &mut Diagnostic,
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
        .map(|&(open, close)| (open.with_hi(close.lo()), same_identation_level(sm, open, close)))
        .collect();

    // sort by `lo`, so the large block spans in the front
    matched_spans.sort_by_key(|(span, _)| span.lo());

    // We use larger block whose identation is well to cover those inner mismatched blocks
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

    // Find the inner-most span candidate for final report
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
            && diag_info.open_braces.last().is_none()
            && diag_info.empty_block_spans.iter().all(|&sp| sp != parent.0.to(parent.1)) {
                err.span_label(parent.0, "this opening brace...");
                err.span_label(parent.1, "...matches this closing brace");
        }
    }
}
