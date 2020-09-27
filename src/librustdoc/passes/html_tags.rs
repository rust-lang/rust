use super::{span_of_attrs, Pass};
use crate::clean::*;
use crate::core::DocContext;
use crate::fold::DocFolder;
use crate::html::markdown::opts;
use core::ops::Range;
use pulldown_cmark::{Event, Parser};
use rustc_feature::UnstableFeatures;
use rustc_session::lint;

pub const CHECK_INVALID_HTML_TAGS: Pass = Pass {
    name: "check-invalid-html-tags",
    run: check_invalid_html_tags,
    description: "detects invalid HTML tags in doc comments",
};

struct InvalidHtmlTagsLinter<'a, 'tcx> {
    cx: &'a DocContext<'tcx>,
}

impl<'a, 'tcx> InvalidHtmlTagsLinter<'a, 'tcx> {
    fn new(cx: &'a DocContext<'tcx>) -> Self {
        InvalidHtmlTagsLinter { cx }
    }
}

pub fn check_invalid_html_tags(krate: Crate, cx: &DocContext<'_>) -> Crate {
    if !UnstableFeatures::from_environment().is_nightly_build() {
        krate
    } else {
        let mut coll = InvalidHtmlTagsLinter::new(cx);

        coll.fold_crate(krate)
    }
}

const ALLOWED_UNCLOSED: &[&str] = &[
    "area", "base", "br", "col", "embed", "hr", "img", "input", "keygen", "link", "meta", "param",
    "source", "track", "wbr",
];

fn drop_tag(
    tags: &mut Vec<(String, Range<usize>)>,
    tag_name: String,
    range: Range<usize>,
    f: &impl Fn(&str, &Range<usize>),
) {
    if let Some(pos) = tags.iter().rev().position(|(t, _)| *t == tag_name) {
        // Because this is from a `rev` iterator, the position is reversed as well!
        let pos = tags.len() - 1 - pos;
        // If the tag is nested inside a "<script>", not warning should be emitted.
        let should_not_warn =
            tags.iter().take(pos + 1).any(|(at, _)| at == "script" || at == "style");
        for (last_tag_name, last_tag_span) in tags.drain(pos + 1..) {
            if should_not_warn || ALLOWED_UNCLOSED.iter().any(|&at| at == &last_tag_name) {
                continue;
            }
            // `tags` is used as a queue, meaning that everything after `pos` is included inside it.
            // So `<h2><h3></h2>` will look like `["h2", "h3"]`. So when closing `h2`, we will still
            // have `h3`, meaning the tag wasn't closed as it should have.
            f(&format!("unclosed HTML tag `{}`", last_tag_name), &last_tag_span);
        }
        // Remove the `tag_name` that was originally closed
        tags.pop();
    } else {
        // It can happen for example in this case: `<h2></script></h2>` (the `h2` tag isn't required
        // but it helps for the visualization).
        f(&format!("unopened HTML tag `{}`", tag_name), &range);
    }
}

fn extract_tag(
    tags: &mut Vec<(String, Range<usize>)>,
    text: &str,
    range: Range<usize>,
    f: &impl Fn(&str, &Range<usize>),
) {
    let mut iter = text.chars().enumerate().peekable();

    while let Some((start_pos, c)) = iter.next() {
        if c == '<' {
            let mut tag_name = String::new();
            let mut is_closing = false;
            while let Some((pos, c)) = iter.peek() {
                // Checking if this is a closing tag (like `</a>` for `<a>`).
                if *c == '/' && tag_name.is_empty() {
                    is_closing = true;
                } else if c.is_ascii_alphanumeric() && !c.is_ascii_uppercase() {
                    tag_name.push(*c);
                } else {
                    if !tag_name.is_empty() {
                        let mut r =
                            Range { start: range.start + start_pos, end: range.start + pos };
                        if *c == '>' {
                            // In case we have a tag without attribute, we can consider the span to
                            // refer to it fully.
                            r.end += 1;
                        }
                        if is_closing {
                            drop_tag(tags, tag_name, r, f);
                        } else {
                            tags.push((tag_name, r));
                        }
                    }
                    break;
                }
                iter.next();
            }
        }
    }
}

impl<'a, 'tcx> DocFolder for InvalidHtmlTagsLinter<'a, 'tcx> {
    fn fold_item(&mut self, item: Item) -> Option<Item> {
        let hir_id = match self.cx.as_local_hir_id(item.def_id) {
            Some(hir_id) => hir_id,
            None => {
                // If non-local, no need to check anything.
                return self.fold_item_recur(item);
            }
        };
        let dox = item.attrs.collapsed_doc_value().unwrap_or_default();
        if !dox.is_empty() {
            let cx = &self.cx;
            let report_diag = |msg: &str, range: &Range<usize>| {
                let sp = match super::source_span_for_markdown_range(cx, &dox, range, &item.attrs) {
                    Some(sp) => sp,
                    None => span_of_attrs(&item.attrs).unwrap_or(item.source.span()),
                };
                cx.tcx.struct_span_lint_hir(lint::builtin::INVALID_HTML_TAGS, hir_id, sp, |lint| {
                    lint.build(msg).emit()
                });
            };

            let mut tags = Vec::new();

            let p = Parser::new_ext(&dox, opts()).into_offset_iter();

            for (event, range) in p {
                match event {
                    Event::Html(text) => extract_tag(&mut tags, &text, range, &report_diag),
                    _ => {}
                }
            }

            for (tag, range) in
                tags.iter().filter(|(t, _)| ALLOWED_UNCLOSED.iter().find(|&at| at == t).is_none())
            {
                report_diag(&format!("unclosed HTML tag `{}`", tag), range);
            }
        }

        self.fold_item_recur(item)
    }
}
