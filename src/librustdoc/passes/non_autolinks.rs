use super::{span_of_attrs, Pass};
use crate::clean::*;
use crate::core::DocContext;
use crate::fold::DocFolder;
use crate::html::markdown::opts;
use core::ops::Range;
use pulldown_cmark::{Event, LinkType, Parser, Tag};
use regex::Regex;
use rustc_errors::Applicability;
use rustc_session::lint;

crate const CHECK_NON_AUTOLINKS: Pass = Pass {
    name: "check-non-autolinks",
    run: check_non_autolinks,
    description: "detects URLS that could be written using angle brackets",
};

const URL_REGEX: &str = concat!(
    r"https?://",                          // url scheme
    r"([-a-zA-Z0-9@:%._\+~#=]{2,256}\.)+", // one or more subdomains
    r"[a-zA-Z]{2,63}",                     // root domain
    r"\b([-a-zA-Z0-9@:%_\+.~#?&/=]*)"      // optional query or url fragments
);

struct NonAutolinksLinter<'a, 'tcx> {
    cx: &'a DocContext<'tcx>,
    regex: Regex,
}

impl<'a, 'tcx> NonAutolinksLinter<'a, 'tcx> {
    fn new(cx: &'a DocContext<'tcx>) -> Self {
        Self { cx, regex: Regex::new(URL_REGEX).expect("failed to build regex") }
    }

    fn find_raw_urls(
        &self,
        text: &str,
        range: Range<usize>,
        f: &impl Fn(&DocContext<'_>, &str, &str, Range<usize>),
    ) {
        // For now, we only check "full" URLs (meaning, starting with "http://" or "https://").
        for match_ in self.regex.find_iter(&text) {
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

crate fn check_non_autolinks(krate: Crate, cx: &DocContext<'_>) -> Crate {
    if !cx.tcx.sess.is_nightly_build() {
        krate
    } else {
        let mut coll = NonAutolinksLinter::new(cx);

        coll.fold_crate(krate)
    }
}

impl<'a, 'tcx> DocFolder for NonAutolinksLinter<'a, 'tcx> {
    fn fold_item(&mut self, item: Item) -> Option<Item> {
        let hir_id = match self.cx.as_local_hir_id(item.def_id) {
            Some(hir_id) => hir_id,
            None => {
                // If non-local, no need to check anything.
                return Some(self.fold_item_recur(item));
            }
        };
        let dox = item.attrs.collapsed_doc_value().unwrap_or_default();
        if !dox.is_empty() {
            let report_diag = |cx: &DocContext<'_>, msg: &str, url: &str, range: Range<usize>| {
                let sp = super::source_span_for_markdown_range(cx, &dox, &range, &item.attrs)
                    .or_else(|| span_of_attrs(&item.attrs))
                    .unwrap_or(item.source.span());
                cx.tcx.struct_span_lint_hir(lint::builtin::NON_AUTOLINKS, hir_id, sp, |lint| {
                    lint.build(msg)
                        .span_suggestion(
                            sp,
                            "use an automatic link instead",
                            format!("<{}>", url),
                            Applicability::MachineApplicable,
                        )
                        .emit()
                });
            };

            let mut p = Parser::new_ext(&dox, opts()).into_offset_iter();

            while let Some((event, range)) = p.next() {
                match event {
                    Event::Start(Tag::Link(kind, _, _)) => {
                        let ignore = matches!(kind, LinkType::Autolink | LinkType::Email);
                        let mut title = String::new();

                        while let Some((event, range)) = p.next() {
                            match event {
                                Event::End(Tag::Link(_, url, _)) => {
                                    // NOTE: links cannot be nested, so we don't need to
                                    // check `kind`
                                    if url.as_ref() == title && !ignore && self.regex.is_match(&url)
                                    {
                                        report_diag(
                                            self.cx,
                                            "unneeded long form for URL",
                                            &url,
                                            range,
                                        );
                                    }
                                    break;
                                }
                                Event::Text(s) if !ignore => title.push_str(&s),
                                _ => {}
                            }
                        }
                    }
                    Event::Text(s) => self.find_raw_urls(&s, range, &report_diag),
                    Event::Start(Tag::CodeBlock(_)) => {
                        // We don't want to check the text inside the code blocks.
                        while let Some((event, _)) = p.next() {
                            match event {
                                Event::End(Tag::CodeBlock(_)) => break,
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
