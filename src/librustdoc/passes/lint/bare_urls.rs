//! Detects links that are not linkified, e.g., in Markdown such as `Go to https://example.com/.`
//! Suggests wrapping the link with angle brackets: `Go to <https://example.com/>.` to linkify it.

use core::ops::Range;
use std::mem;
use std::sync::LazyLock;

use pulldown_cmark::{Event, Parser, Tag};
use regex::Regex;
use rustc_errors::Applicability;
use rustc_hir::HirId;
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
        cx.tcx.node_span_lint(crate::lint::BARE_URLS, hir_id, sp, |lint| {
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
        });
    };

    let mut p = Parser::new_ext(dox, main_body_opts()).into_offset_iter();

    while let Some((event, range)) = p.next() {
        match event {
            Event::Text(s) => find_raw_urls(cx, dox, &s, range, &report_diag),
            // We don't want to check the text inside code blocks or links.
            Event::Start(tag @ (Tag::CodeBlock(_) | Tag::Link { .. })) => {
                for (event, _) in p.by_ref() {
                    match event {
                        Event::End(end)
                            if mem::discriminant(&end) == mem::discriminant(&tag.to_end()) =>
                        {
                            break;
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }
}

static URL_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(concat!(
        r"https?://",                          // url scheme
        r"([-a-zA-Z0-9@:%._\+~#=]{2,256}\.)+", // one or more subdomains
        r"[a-zA-Z]{2,63}",                     // root domain
        r"\b([-a-zA-Z0-9@:%_\+.~#?&/=]*)",     // optional query or url fragments
    ))
    .expect("failed to build regex")
});

fn find_raw_urls(
    cx: &DocContext<'_>,
    dox: &str,
    text: &str,
    range: Range<usize>,
    f: &impl Fn(&DocContext<'_>, &'static str, Range<usize>, Option<&str>),
) {
    trace!("looking for raw urls in {text}");
    // For now, we only check "full" URLs (meaning, starting with "http://" or "https://").
    for match_ in URL_REGEX.find_iter(text) {
        let mut url_range = match_.range();
        url_range.start += range.start;
        url_range.end += range.start;
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
        }
        f(cx, "this URL is not a hyperlink", url_range, without_brackets);
    }
}
