use super::Pass;
use crate::clean::*;
use crate::core::DocContext;
use crate::fold::DocFolder;
use crate::html::markdown::main_body_opts;
use core::ops::Range;
use pulldown_cmark::{Event, Parser, Tag};
use regex::Regex;
use rustc_errors::Applicability;
use std::lazy::SyncLazy;
use std::mem;

crate const CHECK_BARE_URLS: Pass = Pass {
    name: "check-bare-urls",
    run: check_bare_urls,
    description: "detects URLs that are not hyperlinks",
};

static URL_REGEX: SyncLazy<Regex> = SyncLazy::new(|| {
    Regex::new(concat!(
        r"https?://",                          // url scheme
        r"([-a-zA-Z0-9@:%._\+~#=]{2,256}\.)+", // one or more subdomains
        r"[a-zA-Z]{2,63}",                     // root domain
        r"\b([-a-zA-Z0-9@:%_\+.~#?&/=]*)"      // optional query or url fragments
    ))
    .expect("failed to build regex")
});

struct BareUrlsLinter<'a, 'tcx> {
    cx: &'a mut DocContext<'tcx>,
}

impl<'a, 'tcx> BareUrlsLinter<'a, 'tcx> {
    fn find_raw_urls(
        &self,
        text: &str,
        range: Range<usize>,
        f: &impl Fn(&DocContext<'_>, &str, &str, Range<usize>),
    ) {
        trace!("looking for raw urls in {}", text);
        // For now, we only check "full" URLs (meaning, starting with "http://" or "https://").
        for match_ in URL_REGEX.find_iter(text) {
            let url = match_.as_str();
            let url_range = match_.range();
            f(
                self.cx,
                "this URL is not a hyperlink",
                url,
                Range { start: range.start + url_range.start, end: range.start + url_range.end },
            );
        }
    }
}

crate fn check_bare_urls(krate: Crate, cx: &mut DocContext<'_>) -> Crate {
    BareUrlsLinter { cx }.fold_crate(krate)
}

impl<'a, 'tcx> DocFolder for BareUrlsLinter<'a, 'tcx> {
    fn fold_item(&mut self, item: Item) -> Option<Item> {
        let hir_id = match DocContext::as_local_hir_id(self.cx.tcx, item.def_id) {
            Some(hir_id) => hir_id,
            None => {
                // If non-local, no need to check anything.
                return Some(self.fold_item_recur(item));
            }
        };
        let dox = item.attrs.collapsed_doc_value().unwrap_or_default();
        if !dox.is_empty() {
            let report_diag = |cx: &DocContext<'_>, msg: &str, url: &str, range: Range<usize>| {
                let sp = super::source_span_for_markdown_range(cx.tcx, &dox, &range, &item.attrs)
                    .unwrap_or_else(|| item.attr_span(cx.tcx));
                cx.tcx.struct_span_lint_hir(crate::lint::BARE_URLS, hir_id, sp, |lint| {
                    lint.build(msg)
                        .note("bare URLs are not automatically turned into clickable links")
                        .span_suggestion(
                            sp,
                            "use an automatic link instead",
                            format!("<{}>", url),
                            Applicability::MachineApplicable,
                        )
                        .emit()
                });
            };

            let mut p = Parser::new_ext(&dox, main_body_opts()).into_offset_iter();

            while let Some((event, range)) = p.next() {
                match event {
                    Event::Text(s) => self.find_raw_urls(&s, range, &report_diag),
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

        Some(self.fold_item_recur(item))
    }
}
