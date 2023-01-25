//! Detects links that are not linkified, e.g., in Markdown such as `Go to https://example.com/.`
//! Suggests wrapping the link with angle brackets: `Go to <https://example.com/>.` to linkify it.

use crate::clean::*;
use crate::core::DocContext;
use crate::errors::BareUrlNotHyperlink;
use crate::html::markdown::main_body_opts;
use crate::passes::source_span_for_markdown_range;
use core::ops::Range;
use pulldown_cmark::{Event, Parser, Tag};
use regex::Regex;
use std::mem;
use std::sync::LazyLock;

pub(super) fn visit_item(cx: &DocContext<'_>, item: &Item) {
    let Some(hir_id) = DocContext::as_local_hir_id(cx.tcx, item.item_id)
        else {
            // If non-local, no need to check anything.
            return;
        };
    let dox = item.attrs.collapsed_doc_value().unwrap_or_default();
    if !dox.is_empty() {
        let mut p = Parser::new_ext(&dox, main_body_opts()).into_offset_iter();

        while let Some((event, range)) = p.next() {
            match event {
                Event::Text(text) => {
                    trace!("looking for raw urls in {}", text);
                    // For now, we only check "full" URLs (meaning, starting with "http://" or "https://").
                    for match_ in URL_REGEX.find_iter(&text) {
                        let url = match_.as_str();
                        let url_range = match_.range();
                        let range = Range {
                            start: range.start + url_range.start,
                            end: range.start + url_range.end,
                        };
                        let span =
                            source_span_for_markdown_range(cx.tcx, &dox, &range, &item.attrs)
                                .unwrap_or_else(|| item.attr_span(cx.tcx));
                        cx.tcx.emit_spanned_lint(
                            crate::lint::BARE_URLS,
                            hir_id,
                            span,
                            BareUrlNotHyperlink { span, url },
                        );
                    }
                }
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
