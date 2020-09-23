use super::{span_of_attrs, Pass};
use crate::clean::*;
use crate::core::DocContext;
use crate::fold::DocFolder;
use crate::html::markdown::opts;
use pulldown_cmark::{Event, Parser};
use rustc_hir::hir_id::HirId;
use rustc_session::lint;
use rustc_span::Span;

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
    let mut coll = InvalidHtmlTagsLinter::new(cx);

    coll.fold_crate(krate)
}

const ALLOWED_UNCLOSED: &[&str] = &[
    "area", "base", "br", "col", "embed", "hr", "img", "input", "keygen", "link", "meta", "param",
    "source", "track", "wbr",
];

fn drop_tag(
    cx: &DocContext<'_>,
    tags: &mut Vec<String>,
    tag_name: String,
    hir_id: HirId,
    sp: Span,
) {
    if let Some(pos) = tags.iter().position(|t| *t == tag_name) {
        for _ in pos + 1..tags.len() {
            if ALLOWED_UNCLOSED.iter().find(|&at| at == &tags[pos + 1]).is_some() {
                continue;
            }
            // `tags` is used as a queue, meaning that everything after `pos` is included inside it.
            // So `<h2><h3></h2>` will look like `["h2", "h3"]`. So when closing `h2`, we will still
            // have `h3`, meaning the tag wasn't closed as it should have.
            cx.tcx.struct_span_lint_hir(lint::builtin::INVALID_HTML_TAGS, hir_id, sp, |lint| {
                lint.build(&format!("unclosed HTML tag `{}`", tags[pos + 1])).emit()
            });
            tags.remove(pos + 1);
        }
        tags.remove(pos);
    } else {
        // It can happen for example in this case: `<h2></script></h2>` (the `h2` tag isn't required
        // but it helps for the visualization).
        cx.tcx.struct_span_lint_hir(lint::builtin::INVALID_HTML_TAGS, hir_id, sp, |lint| {
            lint.build(&format!("unopened HTML tag `{}`", tag_name)).emit()
        });
    }
}

fn extract_tag(cx: &DocContext<'_>, tags: &mut Vec<String>, text: &str, hir_id: HirId, sp: Span) {
    let mut iter = text.chars().peekable();

    while let Some(c) = iter.next() {
        if c == '<' {
            let mut tag_name = String::new();
            let mut is_closing = false;
            while let Some(&c) = iter.peek() {
                // </tag>
                if c == '/' && tag_name.is_empty() {
                    is_closing = true;
                } else if c.is_ascii_alphanumeric() && !c.is_ascii_uppercase() {
                    tag_name.push(c);
                } else {
                    break;
                }
                iter.next();
            }
            if tag_name.is_empty() {
                // Not an HTML tag presumably...
                continue;
            }
            if is_closing {
                drop_tag(cx, tags, tag_name, hir_id, sp);
            } else {
                tags.push(tag_name);
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
                return None;
            }
        };
        let dox = item.attrs.collapsed_doc_value().unwrap_or_default();
        if !dox.is_empty() {
            let sp = span_of_attrs(&item.attrs).unwrap_or(item.source.span());
            let mut tags = Vec::new();

            let p = Parser::new_ext(&dox, opts());

            for event in p {
                match event {
                    Event::Html(text) => extract_tag(self.cx, &mut tags, &text, hir_id, sp),
                    _ => {}
                }
            }

            for tag in tags.iter().filter(|t| ALLOWED_UNCLOSED.iter().find(|at| at == t).is_none())
            {
                self.cx.tcx.struct_span_lint_hir(
                    lint::builtin::INVALID_HTML_TAGS,
                    hir_id,
                    sp,
                    |lint| lint.build(&format!("unclosed HTML tag `{}`", tag)).emit(),
                );
            }
        }

        self.fold_item_recur(item)
    }
}
