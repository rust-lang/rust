//! Detects links that are not linkified, e.g., in Markdown such as `Go to https://example.com/.`
//! Suggests wrapping the link with angle brackets: `Go to <https://example.com/>.` to linkify it.

use crate::clean::*;
use crate::core::DocContext;
use crate::html::markdown::main_body_opts;
use crate::passes::source_span_for_markdown_range;
use core::ops::Range;
use pulldown_cmark::{Event, Parser, Tag};
use regex::Regex;
use rustc_errors::Applicability;
use std::mem;
use std::sync::LazyLock;

pub(super) fn visit_item(cx: &DocContext<'_>, item: &Item) {
    let Some(hir_id) = DocContext::as_local_hir_id(cx.tcx, item.item_id)
        else {
            // If non-local, no need to check anything.
            return;
        };
    let dox = item.doc_value();
    if !dox.is_empty() {
        let report_diag =
            |cx: &DocContext<'_>, msg: &'static str, url: &str, range: Range<usize>| {
                let sp = source_span_for_markdown_range(cx.tcx, &dox, &range, &item.attrs)
                    .unwrap_or_else(|| item.attr_span(cx.tcx));
                cx.tcx.struct_span_lint_hir(crate::lint::BARE_URLS, hir_id, sp, msg, |lint| {
                    lint.note("bare URLs are not automatically turned into clickable links")
                        .span_suggestion(
                            sp,
                            "use an automatic link instead",
                            format!("<{}>", url),
                            Applicability::MachineApplicable,
                        )
                });
            };

        let mut p = Parser::new_ext(&dox, main_body_opts()).into_offset_iter();

        while let Some((event, range)) = p.next() {
            match event {
                Event::Text(s) => find_raw_urls(cx, &s, range, &report_diag),
                // We don't want to check the text inside code blocks or links.
                Event::Start(tag @ (Tag::CodeBlock(_) | Tag::Link(..))) => {
                    while let Some((event, _)) = p.next() {
                        match event {
                            Event::End(end)
                                if mem::discriminant(&end) == mem::discriminant(&tag) =>
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
}

static URL_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(concat!(
        r"https?://",                          // url scheme
        r"([-a-zA-Z0-9@:%._\+~#=]{2,256}\.)+", // one or more subdomains
        r"[a-zA-Z]{2,63}",                     // root domain
        r"\b([-a-zA-Z0-9@:%_\+.~#?&/=]*)"      // optional query or url fragments
    ))
    .expect("failed to build regex")
});

fn find_raw_urls(
    cx: &DocContext<'_>,
    text: &str,
    range: Range<usize>,
    f: &impl Fn(&DocContext<'_>, &'static str, &str, Range<usize>),
) {
    trace!("looking for raw urls in {}", text);
    // For now, we only check "full" URLs (meaning, starting with "http://" or "https://").
    for match_ in URL_REGEX.find_iter(text) {
        let url = match_.as_str();
        let url_range = match_.range();
        f(
            cx,
            "this URL is not a hyperlink",
            url,
            Range { start: range.start + url_range.start, end: range.start + url_range.end },
        );
    }
}
