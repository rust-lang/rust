//! Detects invalid HTML (like an unclosed `<span>`) in doc comments.

use std::iter::Peekable;
use std::ops::Range;
use std::str::CharIndices;

use pulldown_cmark::{BrokenLink, Event, LinkType, Parser, Tag, TagEnd};
use rustc_hir::HirId;
use rustc_resolve::rustdoc::source_span_for_markdown_range;

use crate::clean::*;
use crate::core::DocContext;
use crate::html::markdown::main_body_opts;

pub(crate) fn visit_item(cx: &DocContext<'_>, item: &Item, hir_id: HirId, dox: &str) {
    let tcx = cx.tcx;
    let report_diag = |msg: String, range: &Range<usize>, is_open_tag: bool| {
        let sp = match source_span_for_markdown_range(tcx, dox, range, &item.attrs.doc_strings) {
            Some(sp) => sp,
            None => item.attr_span(tcx),
        };
        tcx.node_span_lint(crate::lint::INVALID_HTML_TAGS, hir_id, sp, |lint| {
            use rustc_lint_defs::Applicability;

            lint.primary_message(msg);

            // If a tag looks like `<this>`, it might actually be a generic.
            // We don't try to detect stuff `<like, this>` because that's not valid HTML,
            // and we don't try to detect stuff `<like this>` because that's not valid Rust.
            let mut generics_end = range.end;
            if let Some(Some(mut generics_start)) = (is_open_tag
                && dox[..generics_end].ends_with('>'))
            .then(|| extract_path_backwards(dox, range.start))
            {
                while generics_start != 0
                    && generics_end < dox.len()
                    && dox.as_bytes()[generics_start - 1] == b'<'
                    && dox.as_bytes()[generics_end] == b'>'
                {
                    generics_end += 1;
                    generics_start -= 1;
                    if let Some(new_start) = extract_path_backwards(dox, generics_start) {
                        generics_start = new_start;
                    }
                    if let Some(new_end) = extract_path_forward(dox, generics_end) {
                        generics_end = new_end;
                    }
                }
                if let Some(new_end) = extract_path_forward(dox, generics_end) {
                    generics_end = new_end;
                }
                let generics_sp = match source_span_for_markdown_range(
                    tcx,
                    dox,
                    &(generics_start..generics_end),
                    &item.attrs.doc_strings,
                ) {
                    Some(sp) => sp,
                    None => item.attr_span(tcx),
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
                    return;
                }
                // multipart form is chosen here because ``Vec<i32>`` would be confusing.
                lint.multipart_suggestion(
                    "try marking as source code",
                    vec![
                        (generics_sp.shrink_to_lo(), String::from("`")),
                        (generics_sp.shrink_to_hi(), String::from("`")),
                    ],
                    Applicability::MaybeIncorrect,
                );
            }
        });
    };

    let mut tags = Vec::new();
    let mut is_in_comment = None;
    let mut in_code_block = false;

    let link_names = item.link_names(&cx.cache);

    let mut replacer = |broken_link: BrokenLink<'_>| {
        if let Some(link) =
            link_names.iter().find(|link| *link.original_text == *broken_link.reference)
        {
            Some((link.href.as_str().into(), link.new_text.to_string().into()))
        } else if matches!(&broken_link.link_type, LinkType::Reference | LinkType::ReferenceUnknown)
        {
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

    let p = Parser::new_with_broken_link_callback(dox, main_body_opts(), Some(&mut replacer))
        .into_offset_iter();

    for (event, range) in p {
        match event {
            Event::Start(Tag::CodeBlock(_)) => in_code_block = true,
            Event::Html(text) | Event::InlineHtml(text) if !in_code_block => {
                extract_tags(&mut tags, &text, range, &mut is_in_comment, &report_diag)
            }
            Event::End(TagEnd::CodeBlock) => in_code_block = false,
            _ => {}
        }
    }

    for (tag, range) in tags.iter().filter(|(t, _)| {
        let t = t.to_lowercase();
        !ALLOWED_UNCLOSED.contains(&t.as_str())
    }) {
        report_diag(format!("unclosed HTML tag `{tag}`"), range, true);
    }

    if let Some(range) = is_in_comment {
        report_diag("Unclosed HTML comment".to_string(), &range, false);
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
    f: &impl Fn(String, &Range<usize>, bool),
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
            f(format!("unclosed HTML tag `{last_tag_name}`"), &last_tag_span, true);
        }
        // Remove the `tag_name` that was originally closed
        tags.pop();
    } else {
        // It can happen for example in this case: `<h2></script></h2>` (the `h2` tag isn't required
        // but it helps for the visualization).
        f(format!("unopened HTML tag `{tag_name}`"), &range, false);
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
        if let Some(new_pos) = new_pos
            && current_pos != new_pos
        {
            current_pos = new_pos;
            continue;
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
        for c in chars {
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

fn extract_html_tag(
    tags: &mut Vec<(String, Range<usize>)>,
    text: &str,
    range: &Range<usize>,
    start_pos: usize,
    iter: &mut Peekable<CharIndices<'_>>,
    f: &impl Fn(String, &Range<usize>, bool),
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
                    drop_tag(tags, tag_name, r, f);
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
                        f(
                            format!("unclosed quoted HTML attribute on tag `{tag_name}`"),
                            &qr,
                            false,
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
                            f(format!("invalid self-closing HTML tag `{tag_name}`"), &r, false);
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
    tags: &mut Vec<(String, Range<usize>)>,
    text: &str,
    range: Range<usize>,
    is_in_comment: &mut Option<Range<usize>>,
    f: &impl Fn(String, &Range<usize>, bool),
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
