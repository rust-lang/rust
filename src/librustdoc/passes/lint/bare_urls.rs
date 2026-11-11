//! Detects links that are not linkified, e.g., in Markdown such as `Go to https://example.com/.`
//! Suggests wrapping the link with angle brackets: `Go to <https://example.com/>.` to linkify it.

use core::ops::Range;
use std::sync::LazyLock;

use regex::Regex;
use rustc_errors::{Applicability, DiagDecorator};
use rustc_hir::HirId;
use rustc_resolve::rustdoc::pulldown_cmark::{
    DefaultBrokenLinkCallback, Event, Tag, TextMergeWithOffset,
};
use rustc_resolve::rustdoc::source_span_for_markdown_range;
use tracing::trace;

use crate::clean::*;
use crate::core::DocContext;
use crate::html::markdown::main_body_opts;

pub(super) fn visit_item(cx: &DocContext<'_>, item: &Item, hir_id: HirId, dox: &str) {
    let report_diag = |cx: &DocContext<'_>,
                       msg: &'static str,
                       range: Range<usize>,
                       without_brackets: Option<&str>| {
        let maybe_sp = source_span_for_markdown_range(cx.tcx, dox, &range, &item.attrs.doc_strings)
            .map(|(sp, _)| sp);
        let sp = maybe_sp.unwrap_or_else(|| item.attr_span(cx.tcx));
        cx.tcx.emit_node_span_lint(
            crate::lint::BARE_URLS,
            hir_id,
            sp,
            DiagDecorator(|lint| {
                lint.primary_message(msg)
                    .note("bare URLs are not automatically turned into clickable links");
                // The fallback of using the attribute span is suitable for
                // highlighting where the error is, but not for placing the < and >
                if let Some(sp) = maybe_sp {
                    if let Some(without_brackets) = without_brackets {
                        lint.multipart_suggestion(
                            "use an automatic link instead",
                            vec![(sp, format!("<{without_brackets}>"))],
                            Applicability::MachineApplicable,
                        );
                    } else {
                        lint.multipart_suggestion(
                            "use an automatic link instead",
                            vec![
                                (sp.shrink_to_lo(), "<".to_string()),
                                (sp.shrink_to_hi(), ">".to_string()),
                            ],
                            Applicability::MachineApplicable,
                        );
                    }
                }
            }),
        );
    };

    // pulldown-cmark can split a URL into multiple `Text` events while processing
    // characters such as `_` according to CommonMark's emphasis rules.
    // `TextMergeWithOffset` merges these events so we can check the complete URL.
    let mut p = TextMergeWithOffset::<DefaultBrokenLinkCallback>::new_ext(dox, main_body_opts());

    while let Some((event, range)) = p.next() {
        match event {
            Event::Text(_s) => find_raw_urls(cx, dox, range, &report_diag),
            // We don't want to check the text inside code blocks or links.
            Event::Start(tag @ (Tag::CodeBlock(_) | Tag::Link { .. })) => {
                let end = tag.to_end();
                for (event, _) in p.by_ref() {
                    if matches!(event, Event::End(tag) if tag == end) {
                        break;
                    }
                }
            }
            _ => {}
        }
    }
}

static URL_SCHEME_HOST_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(concat!(
        r"https?://",                          // url scheme
        r"([-a-zA-Z0-9@:%._\+~#=]{2,256}\.)+", // one or more subdomains
        r"[a-zA-Z]{2,63}",                     // root domain
    ))
    .expect("failed to build regex")
});

fn find_raw_urls(
    cx: &DocContext<'_>,
    dox: &str,
    range: Range<usize>,
    f: &impl Fn(&DocContext<'_>, &'static str, Range<usize>, Option<&str>),
) {
    trace!("looking for raw urls in {text}", text = &dox[range.clone()]);
    // For now, we only check "full" URLs (meaning, starting with "http://" or "https://").
    for match_ in URL_SCHEME_HOST_REGEX.find_iter(&dox[range.clone()]) {
        let mut url_range = match_.range();
        // We have a range within `dox[range]`.
        // We need a range within `dox` to report the diagnostic.
        url_range.start += range.start;
        url_range.end += range.start;
        // We found the scheme and host. Find the path, query, or fragment.
        // We want to check for matching, balanced parens,
        // but regex isn't powerful enough for that.
        let mut paren_stack = Vec::with_capacity(3);
        'parts: while let Some(&sep) = dox.as_bytes().get(url_range.end) {
            // The hostname must be immediately followed by a path, query,
            // or fragment-declaring separator.
            if !matches!(sep, b'/' | b'?' | b'#') {
                break;
            }
            url_range.end += 1;
            while let Some(&c) = dox.as_bytes().get(url_range.end) {
                if c == b'(' {
                    paren_stack.push(url_range.end);
                } else if c == b')' {
                    // We assume the first unmatched parenthesis marks the end of the url,
                    // as urls rarely contain unbalanced parenthesis in practice.
                    if paren_stack.pop().is_none() {
                        break 'parts;
                    }
                } else if !matches!(
                    c,
                    b'-'
                    | b'a'..=b'z'
                    | b'A'..=b'Z'
                    | b'0'..=b'9'
                    | b'@'
                    | b':'
                    | b'%'
                    | b'_'
                    | b'\\'
                    | b'+'
                    | b'.'
                    | b'~'
                    | b'&'
                    | b'='
                ) {
                    break;
                }
                url_range.end += 1;
            }
        }
        // We assume the first unmatched parenthesis marks the end of the url,
        // as urls rarely contain unbalanced parenthesis in practice.
        if let Some(&end) = paren_stack.first() {
            url_range.end = end;
        }
        let mut without_brackets = None;
        // If the link is contained inside `[]`, then we need to replace the brackets and
        // not just add `<>`.
        if dox[..url_range.start].ends_with('[')
            && url_range.end <= dox.len()
            && dox[url_range.end..].starts_with(']')
        {
            url_range.start -= 1;
            url_range.end += 1;
            without_brackets = Some(match_.as_str());
        } else {
            // Periods are valid in URLs, but very uncommon as the last character of one, while
            // being very common as sentence punctuation right after one. Leave any trailing
            // period out of the link, so that `Visit https://example.com/docs.` is linkified as
            // `Visit <https://example.com/docs>.`.
            let trailing_periods =
                dox[url_range.clone()].len() - dox[url_range.clone()].trim_end_matches('.').len();
            url_range.end -= trailing_periods;
        }
        f(cx, "this URL is not a hyperlink", url_range, without_brackets);
    }
}
