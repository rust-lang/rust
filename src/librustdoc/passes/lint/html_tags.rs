//! Detects invalid HTML (like an unclosed `<span>`) in doc comments.
use crate::clean::*;
use crate::core::DocContext;
use crate::errors::{
    InvalidSelfClosingHtmlTag, MarkSourceCode, UnclosedHtmlComment, UnclosedHtmlTag,
    UnclosedQuotedHtmlAttribute, UnopenedHtmlTag,
};
use crate::html::markdown::main_body_opts;
use crate::passes::source_span_for_markdown_range;

use pulldown_cmark::{BrokenLink, Event, LinkType, Parser, Tag};
use rustc_errors::DecorateLint;
use rustc_hir::HirId;

use std::iter::Peekable;
use std::ops::Range;
use std::str::CharIndices;

pub(crate) fn visit_item(cx: &mut DocContext<'_>, item: &Item) {
    HtmlTagsVisitor { cx }.visit_item(item)
}

const ALLOWED_UNCLOSED: &[&str] = &[
    "area", "base", "br", "col", "embed", "hr", "img", "input", "keygen", "link", "meta", "param",
    "source", "track", "wbr",
];

struct ItemInfo<'a> {
    item: &'a Item,
    hir_id: HirId,
    dox: &'a str,
}

struct HtmlTagsVisitor<'a, 'tcx> {
    cx: &'a mut DocContext<'tcx>,
}

impl<'a, 'tcx> HtmlTagsVisitor<'a, 'tcx> {
    fn visit_item(&self, item: &Item) {
        let tcx = self.cx.tcx;
        // If non-local, no need to check anything.
        let Some(hir_id) = DocContext::as_local_hir_id(tcx, item.item_id) else { return };
        let dox = item.attrs.collapsed_doc_value().unwrap_or_default();
        if !dox.is_empty() {
            let mut tags = Vec::new();
            let mut is_in_comment = None;
            let mut in_code_block = false;

            let link_names = item.link_names(&self.cx.cache);

            let mut replacer = |broken_link: BrokenLink<'_>| {
                if let Some(link) =
                    link_names.iter().find(|link| *link.original_text == *broken_link.reference)
                {
                    Some((link.href.as_str().into(), link.new_text.as_str().into()))
                } else if matches!(
                    &broken_link.link_type,
                    LinkType::Reference | LinkType::ReferenceUnknown
                ) {
                    // If the link is shaped [like][this], suppress any broken HTML in the [this] part.
                    // The `broken_intra_doc_links` will report typos in there anyway.
                    Some((
                        broken_link.reference.to_string().into(),
                        broken_link.reference.to_string().into(),
                    ))
                } else {
                    None
                }
            };

            let p =
                Parser::new_with_broken_link_callback(&dox, main_body_opts(), Some(&mut replacer))
                    .into_offset_iter();

            let item_info = ItemInfo { item, hir_id, dox: &dox };

            for (event, range) in p {
                match event {
                    Event::Start(Tag::CodeBlock(_)) => in_code_block = true,
                    Event::Html(text) if !in_code_block => {
                        self.extract_tags(&mut tags, &text, range, &mut is_in_comment, &item_info)
                    }
                    Event::End(Tag::CodeBlock(_)) => in_code_block = false,
                    _ => {}
                }
            }

            for (tag, range) in tags.iter().filter(|(t, _)| {
                let t = t.to_lowercase();
                !ALLOWED_UNCLOSED.contains(&t.as_str())
            }) {
                self.report_diag(
                    UnclosedHtmlTag {
                        tag,
                        mark_source_code: self.mark_source_code(range, &item_info),
                    },
                    range,
                    &item_info,
                );
            }

            if let Some(range) = is_in_comment {
                self.report_diag(UnclosedHtmlComment, &range, &item_info);
            }
        }
    }

    fn report_diag(
        &self,
        diag: impl for<'b> DecorateLint<'b, ()>,
        range: &Range<usize>,
        item_info: &ItemInfo<'_>,
    ) {
        let sp = match source_span_for_markdown_range(
            self.cx.tcx,
            item_info.dox,
            range,
            &item_info.item.attrs,
        ) {
            Some(sp) => sp,
            None => item_info.item.attr_span(self.cx.tcx),
        };
        self.cx.tcx.emit_spanned_lint(crate::lint::INVALID_HTML_TAGS, item_info.hir_id, sp, diag);
    }

    fn mark_source_code(
        &self,
        range: &Range<usize>,
        item_info: &ItemInfo<'_>,
    ) -> Option<MarkSourceCode> {
        let ItemInfo { item, hir_id: _, dox } = item_info;

        // If a tag looks like `<this>`, it might actually be a generic.
        // We don't try to detect stuff `<like, this>` because that's not valid HTML,
        // and we don't try to detect stuff `<like this>` because that's not valid Rust.
        let mut generics_end = range.end;
        if let Some(Some(mut generics_start)) =
            (dox[..generics_end].ends_with('>')).then(|| extract_path_backwards(&dox, range.start))
        {
            while generics_start != 0
                && generics_end < dox.len()
                && dox.as_bytes()[generics_start - 1] == b'<'
                && dox.as_bytes()[generics_end] == b'>'
            {
                generics_end += 1;
                generics_start -= 1;
                if let Some(new_start) = extract_path_backwards(&dox, generics_start) {
                    generics_start = new_start;
                }
                if let Some(new_end) = extract_path_forward(&dox, generics_end) {
                    generics_end = new_end;
                }
            }
            if let Some(new_end) = extract_path_forward(&dox, generics_end) {
                generics_end = new_end;
            }
            let generics_sp = match source_span_for_markdown_range(
                self.cx.tcx,
                &dox,
                &(generics_start..generics_end),
                &item.attrs,
            ) {
                Some(sp) => sp,
                None => item.attr_span(self.cx.tcx),
            };
            // Sometimes, we only extract part of a path. For example, consider this:
            //
            //     <[u32] as IntoIter<u32>>::Item
            //                       ^^^^^ unclosed HTML tag `u32`
            //
            // We don't have any code for parsing fully-qualified trait paths.
            // In theory, we could add it, but doing it correctly would require
            // parsing the entire path grammar, which is problematic because of
            // overlap between the path grammar and Markdown.
            //
            // The example above shows that ambiguity. Is `[u32]` intended to be an
            // intra-doc link to the u32 primitive, or is it intended to be a slice?
            //
            // If the below conditional were removed, we would suggest this, which is
            // not what the user probably wants.
            //
            //     <[u32] as `IntoIter<u32>`>::Item
            //
            // We know that the user actually wants to wrap the whole thing in a code
            // block, but the only reason we know that is because `u32` does not, in
            // fact, implement IntoIter. If the example looks like this:
            //
            //     <[Vec<i32>] as IntoIter<i32>::Item
            //
            // The ideal fix would be significantly different.
            if (generics_start > 0 && dox.as_bytes()[generics_start - 1] == b'<')
                || (generics_end < dox.len() && dox.as_bytes()[generics_end] == b'>')
            {
                return None;
            }
            // multipart form is chosen here because ``Vec<i32>`` would be confusing.
            Some(MarkSourceCode {
                left: generics_sp.shrink_to_lo(),
                right: generics_sp.shrink_to_hi(),
            })
        } else {
            None
        }
    }

    fn drop_tag(
        &self,
        tags: &mut Vec<(String, Range<usize>)>,
        tag_name: String,
        range: Range<usize>,
        item_info: &ItemInfo<'_>,
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
                self.report_diag(
                    UnclosedHtmlTag {
                        tag: &last_tag_name,
                        mark_source_code: self.mark_source_code(&last_tag_span, item_info),
                    },
                    &last_tag_span,
                    item_info,
                );
            }
            // Remove the `tag_name` that was originally closed
            tags.pop();
        } else {
            // It can happen for example in this case: `<h2></script></h2>` (the `h2` tag isn't required
            // but it helps for the visualization).
            self.report_diag(UnopenedHtmlTag { tag_name }, &range, item_info);
        }
    }

    fn extract_html_tag(
        &self,
        tags: &mut Vec<(String, Range<usize>)>,
        text: &str,
        range: &Range<usize>,
        start_pos: usize,
        iter: &mut Peekable<CharIndices<'_>>,
        item_info: &ItemInfo<'_>,
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
            } else if is_valid_for_html_tag_name(c, tag_name.is_empty()) {
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
                        self.drop_tag(tags, tag_name, r, item_info);
                    } else {
                        let mut is_self_closing = false;
                        let mut quote_pos = None;
                        if c != '>' {
                            let mut quote = None;
                            let mut after_eq = false;
                            for (i, c) in text[pos..].char_indices() {
                                if !c.is_whitespace() {
                                    if let Some(q) = quote {
                                        if c == q {
                                            quote = None;
                                            quote_pos = None;
                                            after_eq = false;
                                        }
                                    } else if c == '>' {
                                        break;
                                    } else if c == '/' && !after_eq {
                                        is_self_closing = true;
                                    } else {
                                        if is_self_closing {
                                            is_self_closing = false;
                                        }
                                        if (c == '"' || c == '\'') && after_eq {
                                            quote = Some(c);
                                            quote_pos = Some(pos + i);
                                        } else if c == '=' {
                                            after_eq = true;
                                        }
                                    }
                                } else if quote.is_none() {
                                    after_eq = false;
                                }
                            }
                        }
                        if let Some(quote_pos) = quote_pos {
                            let qr = Range { start: quote_pos, end: quote_pos };
                            self.report_diag(
                                UnclosedQuotedHtmlAttribute { tag_name: &tag_name },
                                &qr,
                                item_info,
                            );
                        }
                        if is_self_closing {
                            // https://html.spec.whatwg.org/#parse-error-non-void-html-element-start-tag-with-trailing-solidus
                            let valid = ALLOWED_UNCLOSED.contains(&&tag_name[..])
                                || tags.iter().take(pos + 1).any(|(at, _)| {
                                    let at = at.to_lowercase();
                                    at == "svg" || at == "math"
                                });
                            if !valid {
                                self.report_diag(
                                    InvalidSelfClosingHtmlTag { tag_name: &tag_name },
                                    &r,
                                    item_info,
                                );
                            }
                        } else {
                            tags.push((tag_name, r));
                        }
                    }
                }
                break;
            }
            iter.next();
        }
    }

    fn extract_tags(
        &self,
        tags: &mut Vec<(String, Range<usize>)>,
        text: &str,
        range: Range<usize>,
        is_in_comment: &mut Option<Range<usize>>,
        item_info: &ItemInfo<'_>,
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
                    self.extract_html_tag(tags, text, &range, start_pos, &mut iter, item_info);
                }
            }
        }
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
    if current_pos == end_pos { None } else { Some(current_pos) }
}

fn extract_path_forward(text: &str, start_pos: usize) -> Option<usize> {
    use rustc_lexer::{is_id_continue, is_id_start};
    let mut current_pos = start_pos;
    loop {
        if current_pos < text.len() && text[current_pos..].starts_with("::") {
            current_pos += 2;
        } else {
            break;
        }
        let mut chars = text[current_pos..].chars();
        if let Some(c) = chars.next() {
            if is_id_start(c) {
                current_pos += c.len_utf8();
            } else {
                break;
            }
        }
        while let Some(c) = chars.next() {
            if is_id_continue(c) {
                current_pos += c.len_utf8();
            } else {
                break;
            }
        }
    }
    if current_pos == start_pos { None } else { Some(current_pos) }
}

fn is_valid_for_html_tag_name(c: char, is_empty: bool) -> bool {
    // https://spec.commonmark.org/0.30/#raw-html
    //
    // > A tag name consists of an ASCII letter followed by zero or more ASCII letters, digits, or
    // > hyphens (-).
    c.is_ascii_alphabetic() || !is_empty && (c == '-' || c.is_ascii_digit())
}
