//! Detects invalid HTML (like an unclosed `<span>`) in doc comments.
use super::Pass;
use crate::clean::*;
use crate::core::DocContext;
use crate::html::markdown::main_body_opts;
use crate::visit::DocVisitor;

use pulldown_cmark::{Event, Parser, Tag};

use std::iter::Peekable;
use std::ops::Range;
use std::str::CharIndices;

crate const CHECK_INVALID_HTML_TAGS: Pass = Pass {
    name: "check-invalid-html-tags",
    run: check_invalid_html_tags,
    description: "detects invalid HTML tags in doc comments",
};

struct InvalidHtmlTagsLinter<'a, 'tcx> {
    cx: &'a mut DocContext<'tcx>,
}

crate fn check_invalid_html_tags(krate: Crate, cx: &mut DocContext<'_>) -> Crate {
    if cx.tcx.sess.is_nightly_build() {
        let mut coll = InvalidHtmlTagsLinter { cx };
        coll.visit_crate(&krate);
    }
    krate
}

const ALLOWED_UNCLOSED: &[&str] = &[
    "area", "base", "br", "col", "embed", "hr", "img", "input", "keygen", "link", "meta", "param",
    "source", "track", "wbr",
];

fn drop_tag(
    tags: &mut Vec<(String, Range<usize>)>,
    tag_name: String,
    range: Range<usize>,
    f: &impl Fn(&str, &Range<usize>, bool),
) {
    let tag_name_low = tag_name.to_lowercase();
    if let Some(pos) = tags.iter().rposition(|(t, _)| t.to_lowercase() == tag_name_low) {
        // If the tag is nested inside a "<script>" or a "<style>" tag, no warning should
        // be emitted.
        let should_not_warn = tags.iter().take(pos + 1).any(|(at, _)| {
            let at = at.to_lowercase();
            at == "script" || at == "style"
        });
        for (last_tag_name, last_tag_span) in tags.drain(pos + 1..) {
            if should_not_warn {
                continue;
            }
            let last_tag_name_low = last_tag_name.to_lowercase();
            if ALLOWED_UNCLOSED.contains(&last_tag_name_low.as_str()) {
                continue;
            }
            // `tags` is used as a queue, meaning that everything after `pos` is included inside it.
            // So `<h2><h3></h2>` will look like `["h2", "h3"]`. So when closing `h2`, we will still
            // have `h3`, meaning the tag wasn't closed as it should have.
            f(&format!("unclosed HTML tag `{}`", last_tag_name), &last_tag_span, true);
        }
        // Remove the `tag_name` that was originally closed
        tags.pop();
    } else {
        // It can happen for example in this case: `<h2></script></h2>` (the `h2` tag isn't required
        // but it helps for the visualization).
        f(&format!("unopened HTML tag `{}`", tag_name), &range, false);
    }
}

fn extract_path_backwards(text: &str, end_pos: usize) -> Option<usize> {
    use rustc_lexer::{is_id_continue, is_id_start};
    let mut current_pos = end_pos;
    loop {
        if current_pos >= 2 && text[..current_pos].ends_with("::") {
            current_pos -= 2;
        }
        let new_pos = text[..current_pos]
            .char_indices()
            .rev()
            .take_while(|(_, c)| is_id_start(*c) || is_id_continue(*c))
            .reduce(|_accum, item| item)
            .and_then(|(new_pos, c)| is_id_start(c).then_some(new_pos));
        if let Some(new_pos) = new_pos {
            if current_pos != new_pos {
                current_pos = new_pos;
                continue;
            }
        }
        break;
    }
    if current_pos == end_pos {
        return None;
    } else {
        return Some(current_pos);
    }
}

fn extract_html_tag(
    tags: &mut Vec<(String, Range<usize>)>,
    text: &str,
    range: &Range<usize>,
    start_pos: usize,
    iter: &mut Peekable<CharIndices<'_>>,
    f: &impl Fn(&str, &Range<usize>, bool),
) {
    let mut tag_name = String::new();
    let mut is_closing = false;
    let mut prev_pos = start_pos;

    loop {
        let (pos, c) = match iter.peek() {
            Some((pos, c)) => (*pos, *c),
            // In case we reached the of the doc comment, we want to check that it's an
            // unclosed HTML tag. For example "/// <h3".
            None => (prev_pos, '\0'),
        };
        prev_pos = pos;
        // Checking if this is a closing tag (like `</a>` for `<a>`).
        if c == '/' && tag_name.is_empty() {
            is_closing = true;
        } else if c.is_ascii_alphanumeric() {
            tag_name.push(c);
        } else {
            if !tag_name.is_empty() {
                let mut r = Range { start: range.start + start_pos, end: range.start + pos };
                if c == '>' {
                    // In case we have a tag without attribute, we can consider the span to
                    // refer to it fully.
                    r.end += 1;
                }
                if is_closing {
                    // In case we have "</div >" or even "</div         >".
                    if c != '>' {
                        if !c.is_whitespace() {
                            // It seems like it's not a valid HTML tag.
                            break;
                        }
                        let mut found = false;
                        for (new_pos, c) in text[pos..].char_indices() {
                            if !c.is_whitespace() {
                                if c == '>' {
                                    r.end = range.start + new_pos + 1;
                                    found = true;
                                }
                                break;
                            }
                        }
                        if !found {
                            break;
                        }
                    }
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

fn extract_tags(
    tags: &mut Vec<(String, Range<usize>)>,
    text: &str,
    range: Range<usize>,
    is_in_comment: &mut Option<Range<usize>>,
    f: &impl Fn(&str, &Range<usize>, bool),
) {
    let mut iter = text.char_indices().peekable();

    while let Some((start_pos, c)) = iter.next() {
        if is_in_comment.is_some() {
            if text[start_pos..].starts_with("-->") {
                *is_in_comment = None;
            }
        } else if c == '<' {
            if text[start_pos..].starts_with("<!--") {
                // We skip the "!--" part. (Once `advance_by` is stable, might be nice to use it!)
                iter.next();
                iter.next();
                iter.next();
                *is_in_comment = Some(Range {
                    start: range.start + start_pos,
                    end: range.start + start_pos + 3,
                });
            } else {
                extract_html_tag(tags, text, &range, start_pos, &mut iter, f);
            }
        }
    }
}

impl<'a, 'tcx> DocVisitor for InvalidHtmlTagsLinter<'a, 'tcx> {
    fn visit_item(&mut self, item: &Item) {
        let tcx = self.cx.tcx;
        let hir_id = match DocContext::as_local_hir_id(tcx, item.def_id) {
            Some(hir_id) => hir_id,
            None => {
                // If non-local, no need to check anything.
                return;
            }
        };
        let dox = item.attrs.collapsed_doc_value().unwrap_or_default();
        if !dox.is_empty() {
            let report_diag = |msg: &str, range: &Range<usize>, is_open_tag: bool| {
                let sp = match super::source_span_for_markdown_range(tcx, &dox, range, &item.attrs)
                {
                    Some(sp) => sp,
                    None => item.attr_span(tcx),
                };
                tcx.struct_span_lint_hir(crate::lint::INVALID_HTML_TAGS, hir_id, sp, |lint| {
                    use rustc_lint_defs::Applicability;
                    let mut diag = lint.build(msg);
                    // If a tag looks like `<this>`, it might actually be a generic.
                    // We don't try to detect stuff `<like, this>` because that's not valid HTML,
                    // and we don't try to detect stuff `<like this>` because that's not valid Rust.
                    if let Some(Some(generics_start)) = (is_open_tag
                        && dox[..range.end].ends_with(">"))
                    .then(|| extract_path_backwards(&dox, range.start))
                    {
                        let generics_sp = match super::source_span_for_markdown_range(
                            tcx,
                            &dox,
                            &(generics_start..range.end),
                            &item.attrs,
                        ) {
                            Some(sp) => sp,
                            None => item.attr_span(tcx),
                        };
                        // multipart form is chosen here because ``Vec<i32>`` would be confusing.
                        diag.multipart_suggestion(
                            "try marking as source code",
                            vec![
                                (generics_sp.shrink_to_lo(), String::from("`")),
                                (generics_sp.shrink_to_hi(), String::from("`")),
                            ],
                            Applicability::MaybeIncorrect,
                        );
                    }
                    diag.emit()
                });
            };

            let mut tags = Vec::new();
            let mut is_in_comment = None;
            let mut in_code_block = false;

            let p = Parser::new_ext(&dox, main_body_opts()).into_offset_iter();

            for (event, range) in p {
                match event {
                    Event::Start(Tag::CodeBlock(_)) => in_code_block = true,
                    Event::Html(text) | Event::Text(text) if !in_code_block => {
                        extract_tags(&mut tags, &text, range, &mut is_in_comment, &report_diag)
                    }
                    Event::End(Tag::CodeBlock(_)) => in_code_block = false,
                    _ => {}
                }
            }

            for (tag, range) in tags.iter().filter(|(t, _)| {
                let t = t.to_lowercase();
                !ALLOWED_UNCLOSED.contains(&t.as_str())
            }) {
                report_diag(&format!("unclosed HTML tag `{}`", tag), range, true);
            }

            if let Some(range) = is_in_comment {
                report_diag("Unclosed HTML comment", &range, false);
            }
        }

        self.visit_item_recur(item)
    }
}
