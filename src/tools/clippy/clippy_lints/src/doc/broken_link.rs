use clippy_utils::diagnostics::span_lint;
use pulldown_cmark::BrokenLink as PullDownBrokenLink;
use rustc_lint::LateContext;
use rustc_resolve::rustdoc::{DocFragment, source_span_for_markdown_range};
use rustc_span::{BytePos, Pos, Span};

use super::DOC_BROKEN_LINK;

/// Scan and report broken link on documents.
/// It ignores false positives detected by `pulldown_cmark`, and only
/// warns users when the broken link is consider a URL.
// NOTE: We don't check these other cases because
// rustdoc itself will check and warn about it:
// - When a link url is broken across multiple lines in the URL path part
// - When a link tag is missing the close parenthesis character at the end.
// - When a link has whitespace within the url link.
pub fn check(cx: &LateContext<'_>, bl: &PullDownBrokenLink<'_>, doc: &str, fragments: &[DocFragment]) {
    warn_if_broken_link(cx, bl, doc, fragments);
}

fn warn_if_broken_link(cx: &LateContext<'_>, bl: &PullDownBrokenLink<'_>, doc: &str, fragments: &[DocFragment]) {
    let mut len = 0;

    // grab raw link data
    let (_, raw_link) = doc.split_at(bl.span.start);

    // strip off link text part
    let raw_link = match raw_link.split_once(']') {
        None => return,
        Some((prefix, suffix)) => {
            len += prefix.len() + 1;
            suffix
        },
    };

    let raw_link = match raw_link.split_once('(') {
        None => return,
        Some((prefix, suffix)) => {
            if !prefix.is_empty() {
                // there is text between ']' and '(' chars, so it is not a valid link
                return;
            }
            len += prefix.len() + 1;
            suffix
        },
    };

    if raw_link.starts_with("(http") {
        // reduce chances of false positive reports
        // by limiting this checking only to http/https links.
        return;
    }

    for c in raw_link.chars() {
        if c == ')' {
            // it is a valid link
            return;
        }

        if c == '\n'
            && let Some((span, _)) = source_span_for_markdown_range(cx.tcx, doc, &bl.span, fragments)
        {
            report_broken_link(cx, span, len);
            break;
        }

        len += 1;
    }
}

fn report_broken_link(cx: &LateContext<'_>, frag_span: Span, offset: usize) {
    let start = frag_span.lo();
    let end = start + BytePos::from_usize(offset);

    let span = Span::new(start, end, frag_span.ctxt(), frag_span.parent());

    span_lint(
        cx,
        DOC_BROKEN_LINK,
        span,
        "possible broken doc link: broken across multiple lines",
    );
}
