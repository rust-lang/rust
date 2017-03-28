// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Markdown formatting for rustdoc
//!
//! This module implements markdown formatting through the pulldown-cmark
//! rust-library. This module exposes all of the
//! functionality through a unit-struct, `Markdown`, which has an implementation
//! of `fmt::Display`. Example usage:
//!
//! ```rust,ignore
//! use rustdoc::html::markdown::{Markdown, MarkdownOutputStyle};
//!
//! let s = "My *markdown* _text_";
//! let html = format!("{}", Markdown(s, MarkdownOutputStyle::Fancy));
//! // ... something using html
//! ```

#![allow(non_camel_case_types)]

use std::ascii::AsciiExt;
use std::cell::RefCell;
use std::default::Default;
use std::fmt::{self, Write};
use std::str;
use syntax::feature_gate::UnstableFeatures;
use syntax::codemap::Span;

use html::render::derive_id;
use html::toc::TocBuilder;
use html::highlight;
use html::escape::Escape;
use test;

use pulldown_cmark::{self, Event, Parser, Tag};

#[derive(Copy, Clone)]
pub enum MarkdownOutputStyle {
    Compact,
    Fancy,
}

impl MarkdownOutputStyle {
    pub fn is_compact(&self) -> bool {
        match *self {
            MarkdownOutputStyle::Compact => true,
            _ => false,
        }
    }

    pub fn is_fancy(&self) -> bool {
        match *self {
            MarkdownOutputStyle::Fancy => true,
            _ => false,
        }
    }
}

/// A unit struct which has the `fmt::Display` trait implemented. When
/// formatted, this struct will emit the HTML corresponding to the rendered
/// version of the contained markdown string.
// The second parameter is whether we need a shorter version or not.
pub struct Markdown<'a>(pub &'a str, pub MarkdownOutputStyle);
/// A unit struct like `Markdown`, that renders the markdown with a
/// table of contents.
pub struct MarkdownWithToc<'a>(pub &'a str);
/// A unit struct like `Markdown`, that renders the markdown escaping HTML tags.
pub struct MarkdownHtml<'a>(pub &'a str);

/// Returns Some(code) if `s` is a line that should be stripped from
/// documentation but used in example code. `code` is the portion of
/// `s` that should be used in tests. (None for lines that should be
/// left as-is.)
fn stripped_filtered_line<'a>(s: &'a str) -> Option<&'a str> {
    let trimmed = s.trim();
    if trimmed == "#" {
        Some("")
    } else if trimmed.starts_with("# ") {
        Some(&trimmed[2..])
    } else {
        None
    }
}

/// Returns a new string with all consecutive whitespace collapsed into
/// single spaces.
///
/// Any leading or trailing whitespace will be trimmed.
fn collapse_whitespace(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

// Information about the playground if a URL has been specified, containing an
// optional crate name and the URL.
thread_local!(pub static PLAYGROUND: RefCell<Option<(Option<String>, String)>> = {
    RefCell::new(None)
});

macro_rules! event_loop_break {
    ($parser:expr, $toc_builder:expr, $shorter:expr, $buf:expr, $escape:expr, $id:expr,
     $($end_event:pat)|*) => {{
        fn inner(id: &mut Option<&mut String>, s: &str) {
            if let Some(ref mut id) = *id {
                id.push_str(s);
            }
        }
        while let Some(event) = $parser.next() {
            match event {
                $($end_event)|* => break,
                Event::Text(ref s) => {
                    inner($id, s);
                    if $escape {
                        $buf.push_str(&format!("{}", Escape(s)));
                    } else {
                        $buf.push_str(s);
                    }
                }
                Event::SoftBreak | Event::HardBreak if !$buf.is_empty() => {
                    $buf.push(' ');
                }
                x => {
                    looper($parser, &mut $buf, Some(x), $toc_builder, $shorter, $id);
                }
            }
        }
    }}
}

pub fn render(w: &mut fmt::Formatter,
              s: &str,
              print_toc: bool,
              shorter: MarkdownOutputStyle) -> fmt::Result {
    fn code_block(parser: &mut Parser, buffer: &mut String, lang: &str) {
        let mut origtext = String::new();
        while let Some(event) = parser.next() {
            match event {
                Event::End(Tag::CodeBlock(_)) => break,
                Event::Text(ref s) => {
                    origtext.push_str(s);
                }
                _ => {}
            }
        }
        let origtext = origtext.trim_left();
        debug!("docblock: ==============\n{:?}\n=======", origtext);

        let lines = origtext.lines().filter(|l| {
            stripped_filtered_line(*l).is_none()
        });
        let text = lines.collect::<Vec<&str>>().join("\n");
        let block_info = if lang.is_empty() {
            LangString::all_false()
        } else {
            LangString::parse(lang)
        };
        if !block_info.rust {
            buffer.push_str(&format!("<pre><code class=\"language-{}\">{}</code></pre>",
                            lang, text));
            return
        }
        PLAYGROUND.with(|play| {
            // insert newline to clearly separate it from the
            // previous block so we can shorten the html output
            buffer.push('\n');
            let playground_button = play.borrow().as_ref().and_then(|&(ref krate, ref url)| {
                if url.is_empty() {
                    return None;
                }
                let test = origtext.lines().map(|l| {
                    stripped_filtered_line(l).unwrap_or(l)
                }).collect::<Vec<&str>>().join("\n");
                let krate = krate.as_ref().map(|s| &**s);
                let test = test::maketest(&test, krate, false,
                                          &Default::default());
                let channel = if test.contains("#![feature(") {
                    "&amp;version=nightly"
                } else {
                    ""
                };
                // These characters don't need to be escaped in a URI.
                // FIXME: use a library function for percent encoding.
                fn dont_escape(c: u8) -> bool {
                    (b'a' <= c && c <= b'z') ||
                    (b'A' <= c && c <= b'Z') ||
                    (b'0' <= c && c <= b'9') ||
                    c == b'-' || c == b'_' || c == b'.' ||
                    c == b'~' || c == b'!' || c == b'\'' ||
                    c == b'(' || c == b')' || c == b'*'
                }
                let mut test_escaped = String::new();
                for b in test.bytes() {
                    if dont_escape(b) {
                        test_escaped.push(char::from(b));
                    } else {
                        write!(test_escaped, "%{:02X}", b).unwrap();
                    }
                }
                Some(format!(
                    r#"<a class="test-arrow" target="_blank" href="{}?code={}{}">Run</a>"#,
                    url, test_escaped, channel
                ))
            });
            buffer.push_str(&highlight::render_with_highlighting(
                            &text,
                            Some("rust-example-rendered"),
                            None,
                            playground_button.as_ref().map(String::as_str)));
        });
    }

    fn heading(parser: &mut Parser, buffer: &mut String, toc_builder: &mut Option<TocBuilder>,
               shorter: MarkdownOutputStyle, level: i32) {
        let mut ret = String::new();
        let mut id = String::new();
        event_loop_break!(parser, toc_builder, shorter, ret, true, &mut Some(&mut id),
                          Event::End(Tag::Header(_)));
        ret = ret.trim_right().to_owned();

        let id = id.chars().filter_map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '_' {
                if c.is_ascii() {
                    Some(c.to_ascii_lowercase())
                } else {
                    Some(c)
                }
            } else if c.is_whitespace() && c.is_ascii() {
                Some('-')
            } else {
                None
            }
        }).collect::<String>();

        let id = derive_id(id);

        let sec = toc_builder.as_mut().map_or("".to_owned(), |builder| {
            format!("{} ", builder.push(level as u32, ret.clone(), id.clone()))
        });

        // Render the HTML
        buffer.push_str(&format!("<h{lvl} id=\"{id}\" class=\"section-header\">\
                                  <a href=\"#{id}\">{sec}{}</a></h{lvl}>",
                                 ret, lvl = level, id = id, sec = sec));
    }

    fn inline_code(parser: &mut Parser, buffer: &mut String, toc_builder: &mut Option<TocBuilder>,
                   shorter: MarkdownOutputStyle, id: &mut Option<&mut String>) {
        let mut content = String::new();
        event_loop_break!(parser, toc_builder, shorter, content, false, id, Event::End(Tag::Code));
        buffer.push_str(&format!("<code>{}</code>",
                                 Escape(&collapse_whitespace(content.trim_right()))));
    }

    fn link(parser: &mut Parser, buffer: &mut String, toc_builder: &mut Option<TocBuilder>,
            shorter: MarkdownOutputStyle, url: &str, mut title: String,
            id: &mut Option<&mut String>) {
        event_loop_break!(parser, toc_builder, shorter, title, true, id,
                          Event::End(Tag::Link(_, _)));
        buffer.push_str(&format!("<a href=\"{}\">{}</a>", url, title));
    }

    fn paragraph(parser: &mut Parser, buffer: &mut String, toc_builder: &mut Option<TocBuilder>,
                 shorter: MarkdownOutputStyle, id: &mut Option<&mut String>) {
        let mut content = String::new();
        event_loop_break!(parser, toc_builder, shorter, content, true, id,
                          Event::End(Tag::Paragraph));
        buffer.push_str(&format!("<p>{}</p>", content.trim_right()));
    }

    fn table_cell(parser: &mut Parser, buffer: &mut String, toc_builder: &mut Option<TocBuilder>,
                  shorter: MarkdownOutputStyle) {
        let mut content = String::new();
        event_loop_break!(parser, toc_builder, shorter, content, true, &mut None,
                          Event::End(Tag::TableHead) |
                              Event::End(Tag::Table(_)) |
                              Event::End(Tag::TableRow) |
                              Event::End(Tag::TableCell));
        buffer.push_str(&format!("<td>{}</td>", content.trim()));
    }

    fn table_row(parser: &mut Parser, buffer: &mut String, toc_builder: &mut Option<TocBuilder>,
                 shorter: MarkdownOutputStyle) {
        let mut content = String::new();
        while let Some(event) = parser.next() {
            match event {
                Event::End(Tag::TableHead) |
                    Event::End(Tag::Table(_)) |
                    Event::End(Tag::TableRow) => break,
                Event::Start(Tag::TableCell) => {
                    table_cell(parser, &mut content, toc_builder, shorter);
                }
                x => {
                    looper(parser, &mut content, Some(x), toc_builder, shorter, &mut None);
                }
            }
        }
        buffer.push_str(&format!("<tr>{}</tr>", content));
    }

    fn table_head(parser: &mut Parser, buffer: &mut String, toc_builder: &mut Option<TocBuilder>,
                  shorter: MarkdownOutputStyle) {
        let mut content = String::new();
        while let Some(event) = parser.next() {
            match event {
                Event::End(Tag::TableHead) | Event::End(Tag::Table(_)) => break,
                Event::Start(Tag::TableCell) => {
                    table_cell(parser, &mut content, toc_builder, shorter);
                }
                x => {
                    looper(parser, &mut content, Some(x), toc_builder, shorter, &mut None);
                }
            }
        }
        if !content.is_empty() {
            buffer.push_str(&format!("<thead><tr>{}</tr></thead>", content.replace("td>", "th>")));
        }
    }

    fn table(parser: &mut Parser, buffer: &mut String, toc_builder: &mut Option<TocBuilder>,
             shorter: MarkdownOutputStyle) {
        let mut content = String::new();
        let mut rows = String::new();
        while let Some(event) = parser.next() {
            match event {
                Event::End(Tag::Table(_)) => break,
                Event::Start(Tag::TableHead) => {
                    table_head(parser, &mut content, toc_builder, shorter);
                }
                Event::Start(Tag::TableRow) => {
                    table_row(parser, &mut rows, toc_builder, shorter);
                }
                _ => {}
            }
        }
        buffer.push_str(&format!("<table>{}{}</table>",
                                 content,
                                 if shorter.is_compact() || rows.is_empty() {
                                     String::new()
                                 } else {
                                     format!("<tbody>{}</tbody>", rows)
                                 }));
    }

    fn blockquote(parser: &mut Parser, buffer: &mut String, toc_builder: &mut Option<TocBuilder>,
                  shorter: MarkdownOutputStyle) {
        let mut content = String::new();
        event_loop_break!(parser, toc_builder, shorter, content, true, &mut None,
                          Event::End(Tag::BlockQuote));
        buffer.push_str(&format!("<blockquote>{}</blockquote>", content.trim_right()));
    }

    fn list_item(parser: &mut Parser, buffer: &mut String, toc_builder: &mut Option<TocBuilder>,
                 shorter: MarkdownOutputStyle) {
        let mut content = String::new();
        while let Some(event) = parser.next() {
            match event {
                Event::End(Tag::Item) => break,
                Event::Text(ref s) => {
                    content.push_str(&format!("{}", Escape(s)));
                }
                x => {
                    looper(parser, &mut content, Some(x), toc_builder, shorter, &mut None);
                }
            }
        }
        buffer.push_str(&format!("<li>{}</li>", content));
    }

    fn list(parser: &mut Parser, buffer: &mut String, toc_builder: &mut Option<TocBuilder>,
            shorter: MarkdownOutputStyle) {
        let mut content = String::new();
        while let Some(event) = parser.next() {
            match event {
                Event::End(Tag::List(_)) => break,
                Event::Start(Tag::Item) => {
                    list_item(parser, &mut content, toc_builder, shorter);
                }
                x => {
                    looper(parser, &mut content, Some(x), toc_builder, shorter, &mut None);
                }
            }
        }
        buffer.push_str(&format!("<ul>{}</ul>", content));
    }

    fn emphasis(parser: &mut Parser, buffer: &mut String, toc_builder: &mut Option<TocBuilder>,
                shorter: MarkdownOutputStyle, id: &mut Option<&mut String>) {
        let mut content = String::new();
        event_loop_break!(parser, toc_builder, shorter, content, false, id,
                          Event::End(Tag::Emphasis));
        buffer.push_str(&format!("<em>{}</em>", content));
    }

    fn strong(parser: &mut Parser, buffer: &mut String, toc_builder: &mut Option<TocBuilder>,
              shorter: MarkdownOutputStyle, id: &mut Option<&mut String>) {
        let mut content = String::new();
        event_loop_break!(parser, toc_builder, shorter, content, false, id,
                          Event::End(Tag::Strong));
        buffer.push_str(&format!("<strong>{}</strong>", content));
    }

    fn looper<'a>(parser: &'a mut Parser, buffer: &mut String, next_event: Option<Event<'a>>,
                  toc_builder: &mut Option<TocBuilder>, shorter: MarkdownOutputStyle,
                  id: &mut Option<&mut String>) -> bool {
        if let Some(event) = next_event {
            match event {
                Event::Start(Tag::CodeBlock(lang)) => {
                    code_block(parser, buffer, &*lang);
                }
                Event::Start(Tag::Header(level)) => {
                    heading(parser, buffer, toc_builder, shorter, level);
                }
                Event::Start(Tag::Code) => {
                    inline_code(parser, buffer, toc_builder, shorter, id);
                }
                Event::Start(Tag::Paragraph) => {
                    paragraph(parser, buffer, toc_builder, shorter, id);
                }
                Event::Start(Tag::Link(ref url, ref t)) => {
                    link(parser, buffer, toc_builder, shorter, url, t.as_ref().to_owned(), id);
                }
                Event::Start(Tag::Table(_)) => {
                    table(parser, buffer, toc_builder, shorter);
                }
                Event::Start(Tag::BlockQuote) => {
                    blockquote(parser, buffer, toc_builder, shorter);
                }
                Event::Start(Tag::List(_)) => {
                    list(parser, buffer, toc_builder, shorter);
                }
                Event::Start(Tag::Emphasis) => {
                    emphasis(parser, buffer, toc_builder, shorter, id);
                }
                Event::Start(Tag::Strong) => {
                    strong(parser, buffer, toc_builder, shorter, id);
                }
                Event::Html(h) | Event::InlineHtml(h) => {
                    buffer.push_str(&*h);
                }
                _ => {}
            }
            shorter.is_fancy()
        } else {
            false
        }
    }

    let mut toc_builder = if print_toc {
        Some(TocBuilder::new())
    } else {
        None
    };
    let mut buffer = String::new();
    let mut parser = Parser::new_ext(s, pulldown_cmark::OPTION_ENABLE_TABLES);
    loop {
        let next_event = parser.next();
        if !looper(&mut parser, &mut buffer, next_event, &mut toc_builder, shorter, &mut None) {
            break
        }
    }
    let mut ret = toc_builder.map_or(Ok(()), |builder| {
        write!(w, "<nav id=\"TOC\">{}</nav>", builder.into_toc())
    });

    if ret.is_ok() {
        ret = w.write_str(&buffer);
    }
    ret
}

pub fn find_testable_code(doc: &str, tests: &mut ::test::Collector, position: Span) {
    tests.set_position(position);

    let mut parser = Parser::new(doc);
    let mut prev_offset = 0;
    let mut nb_lines = 0;
    let mut register_header = None;
    'main: while let Some(event) = parser.next() {
        match event {
            Event::Start(Tag::CodeBlock(s)) => {
                let block_info = if s.is_empty() {
                    LangString::all_false()
                } else {
                    LangString::parse(&*s)
                };
                if !block_info.rust {
                    continue
                }
                let mut test_s = String::new();
                let mut offset = None;
                loop {
                    let event = parser.next();
                    if let Some(event) = event {
                        match event {
                            Event::End(Tag::CodeBlock(_)) => break,
                            Event::Text(ref s) => {
                                test_s.push_str(s);
                                if offset.is_none() {
                                    offset = Some(parser.get_offset());
                                }
                            }
                            _ => {}
                        }
                    } else {
                        break 'main;
                    }
                }
                let offset = offset.unwrap_or(0);
                let lines = test_s.lines().map(|l| {
                    stripped_filtered_line(l).unwrap_or(l)
                });
                let text = lines.collect::<Vec<&str>>().join("\n");
                nb_lines += doc[prev_offset..offset].lines().count();
                let line = tests.get_line() + (nb_lines - 1);
                let filename = tests.get_filename();
                tests.add_test(text.to_owned(),
                               block_info.should_panic, block_info.no_run,
                               block_info.ignore, block_info.test_harness,
                               block_info.compile_fail, block_info.error_codes,
                               line, filename);
                prev_offset = offset;
            }
            Event::Start(Tag::Header(level)) => {
                register_header = Some(level as u32);
            }
            Event::Text(ref s) if register_header.is_some() => {
                let level = register_header.unwrap();
                if s.is_empty() {
                    tests.register_header("", level);
                } else {
                    tests.register_header(s, level);
                }
                register_header = None;
            }
            _ => {}
        }
    }
}

#[derive(Eq, PartialEq, Clone, Debug)]
struct LangString {
    original: String,
    should_panic: bool,
    no_run: bool,
    ignore: bool,
    rust: bool,
    test_harness: bool,
    compile_fail: bool,
    error_codes: Vec<String>,
}

impl LangString {
    fn all_false() -> LangString {
        LangString {
            original: String::new(),
            should_panic: false,
            no_run: false,
            ignore: false,
            rust: true,  // NB This used to be `notrust = false`
            test_harness: false,
            compile_fail: false,
            error_codes: Vec::new(),
        }
    }

    fn parse(string: &str) -> LangString {
        let mut seen_rust_tags = false;
        let mut seen_other_tags = false;
        let mut data = LangString::all_false();
        let mut allow_compile_fail = false;
        let mut allow_error_code_check = false;
        if UnstableFeatures::from_environment().is_nightly_build() {
            allow_compile_fail = true;
            allow_error_code_check = true;
        }

        data.original = string.to_owned();
        let tokens = string.split(|c: char|
            !(c == '_' || c == '-' || c.is_alphanumeric())
        );

        for token in tokens {
            match token {
                "" => {},
                "should_panic" => { data.should_panic = true; seen_rust_tags = true; },
                "no_run" => { data.no_run = true; seen_rust_tags = true; },
                "ignore" => { data.ignore = true; seen_rust_tags = true; },
                "rust" => { data.rust = true; seen_rust_tags = true; },
                "test_harness" => { data.test_harness = true; seen_rust_tags = true; },
                "compile_fail" if allow_compile_fail => {
                    data.compile_fail = true;
                    seen_rust_tags = true;
                    data.no_run = true;
                }
                x if allow_error_code_check && x.starts_with("E") && x.len() == 5 => {
                    if let Ok(_) = x[1..].parse::<u32>() {
                        data.error_codes.push(x.to_owned());
                        seen_rust_tags = true;
                    } else {
                        seen_other_tags = true;
                    }
                }
                _ => { seen_other_tags = true }
            }
        }

        data.rust &= !seen_other_tags || seen_rust_tags;

        data
    }
}

impl<'a> fmt::Display for Markdown<'a> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        let Markdown(md, shorter) = *self;
        // This is actually common enough to special-case
        if md.is_empty() { return Ok(()) }
        render(fmt, md, false, shorter)
    }
}

impl<'a> fmt::Display for MarkdownWithToc<'a> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        let MarkdownWithToc(md) = *self;
        render(fmt, md, true, MarkdownOutputStyle::Fancy)
    }
}

impl<'a> fmt::Display for MarkdownHtml<'a> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        let MarkdownHtml(md) = *self;
        // This is actually common enough to special-case
        if md.is_empty() { return Ok(()) }
        render(fmt, md, false, MarkdownOutputStyle::Fancy)
    }
}

pub fn plain_summary_line(md: &str) -> String {
    struct ParserWrapper<'a> {
        inner: Parser<'a>,
        is_in: isize,
        is_first: bool,
    }

    impl<'a> Iterator for ParserWrapper<'a> {
        type Item = String;

        fn next(&mut self) -> Option<String> {
            let next_event = self.inner.next();
            if next_event.is_none() {
                return None
            }
            let next_event = next_event.unwrap();
            let (ret, is_in) = match next_event {
                Event::Start(Tag::Paragraph) => (None, 1),
                Event::Start(Tag::Link(_, ref t)) if !self.is_first => {
                    (Some(t.as_ref().to_owned()), 1)
                }
                Event::Start(Tag::Code) => (Some("`".to_owned()), 1),
                Event::End(Tag::Code) => (Some("`".to_owned()), -1),
                Event::Start(Tag::Header(_)) => (None, 1),
                Event::Text(ref s) if self.is_in > 0 => (Some(s.as_ref().to_owned()), 0),
                Event::End(Tag::Link(_, ref t)) => (Some(t.as_ref().to_owned()), -1),
                Event::End(Tag::Paragraph) | Event::End(Tag::Header(_)) => (None, -1),
                _ => (None, 0),
            };
            if is_in > 0 || (is_in < 0 && self.is_in > 0) {
                self.is_in += is_in;
            }
            if ret.is_some() {
                self.is_first = false;
                ret
            } else {
                Some(String::new())
            }
        }
    }
    let mut s = String::with_capacity(md.len() * 3 / 2);
    let mut p = ParserWrapper {
        inner: Parser::new(md),
        is_in: 0,
        is_first: true,
    };
    while let Some(t) = p.next() {
        if !t.is_empty() {
            s.push_str(&t);
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::{LangString, Markdown, MarkdownHtml, MarkdownOutputStyle};
    use super::plain_summary_line;
    use html::render::reset_ids;

    #[test]
    fn test_lang_string_parse() {
        fn t(s: &str,
            should_panic: bool, no_run: bool, ignore: bool, rust: bool, test_harness: bool,
            compile_fail: bool, error_codes: Vec<String>) {
            assert_eq!(LangString::parse(s), LangString {
                should_panic: should_panic,
                no_run: no_run,
                ignore: ignore,
                rust: rust,
                test_harness: test_harness,
                compile_fail: compile_fail,
                error_codes: error_codes,
                original: s.to_owned(),
            })
        }

        // marker                | should_panic| no_run| ignore| rust | test_harness| compile_fail
        //                       | error_codes
        t("",                      false,        false,  false,  true,  false, false, Vec::new());
        t("rust",                  false,        false,  false,  true,  false, false, Vec::new());
        t("sh",                    false,        false,  false,  false, false, false, Vec::new());
        t("ignore",                false,        false,  true,   true,  false, false, Vec::new());
        t("should_panic",          true,         false,  false,  true,  false, false, Vec::new());
        t("no_run",                false,        true,   false,  true,  false, false, Vec::new());
        t("test_harness",          false,        false,  false,  true,  true,  false, Vec::new());
        t("compile_fail",          false,        true,   false,  true,  false, true,  Vec::new());
        t("{.no_run .example}",    false,        true,   false,  true,  false, false, Vec::new());
        t("{.sh .should_panic}",   true,         false,  false,  true,  false, false, Vec::new());
        t("{.example .rust}",      false,        false,  false,  true,  false, false, Vec::new());
        t("{.test_harness .rust}", false,        false,  false,  true,  true,  false, Vec::new());
    }

    #[test]
    fn issue_17736() {
        let markdown = "# title";
        format!("{}", Markdown(markdown, MarkdownOutputStyle::Fancy));
        reset_ids(true);
    }

    #[test]
    fn test_header() {
        fn t(input: &str, expect: &str) {
            let output = format!("{}", Markdown(input, MarkdownOutputStyle::Fancy));
            assert_eq!(output, expect, "original: {}", input);
            reset_ids(true);
        }

        t("# Foo bar", "<h1 id=\"foo-bar\" class=\"section-header\">\
          <a href=\"#foo-bar\">Foo bar</a></h1>");
        t("## Foo-bar_baz qux", "<h2 id=\"foo-bar_baz-qux\" class=\"section-\
          header\"><a href=\"#foo-bar_baz-qux\">Foo-bar_baz qux</a></h2>");
        t("### **Foo** *bar* baz!?!& -_qux_-%",
          "<h3 id=\"foo-bar-baz--qux-\" class=\"section-header\">\
          <a href=\"#foo-bar-baz--qux-\"><strong>Foo</strong> \
          <em>bar</em> baz!?!&amp; -<em>qux</em>-%</a></h3>");
        t("#### **Foo?** & \\*bar?!*  _`baz`_ ❤ #qux",
          "<h4 id=\"foo--bar--baz--qux\" class=\"section-header\">\
          <a href=\"#foo--bar--baz--qux\"><strong>Foo?</strong> &amp; *bar?!*  \
          <em><code>baz</code></em> ❤ #qux</a></h4>");
    }

    #[test]
    fn test_header_ids_multiple_blocks() {
        fn t(input: &str, expect: &str) {
            let output = format!("{}", Markdown(input, MarkdownOutputStyle::Fancy));
            assert_eq!(output, expect, "original: {}", input);
        }

        let test = || {
            t("# Example", "<h1 id=\"example\" class=\"section-header\">\
              <a href=\"#example\">Example</a></h1>");
            t("# Panics", "<h1 id=\"panics\" class=\"section-header\">\
              <a href=\"#panics\">Panics</a></h1>");
            t("# Example", "<h1 id=\"example-1\" class=\"section-header\">\
              <a href=\"#example-1\">Example</a></h1>");
            t("# Main", "<h1 id=\"main-1\" class=\"section-header\">\
              <a href=\"#main-1\">Main</a></h1>");
            t("# Example", "<h1 id=\"example-2\" class=\"section-header\">\
              <a href=\"#example-2\">Example</a></h1>");
            t("# Panics", "<h1 id=\"panics-1\" class=\"section-header\">\
              <a href=\"#panics-1\">Panics</a></h1>");
        };
        test();
        reset_ids(true);
        test();
    }

    #[test]
    fn test_plain_summary_line() {
        fn t(input: &str, expect: &str) {
            let output = plain_summary_line(input);
            assert_eq!(output, expect, "original: {}", input);
        }

        t("hello [Rust](https://www.rust-lang.org) :)", "hello Rust :)");
        t("code `let x = i32;` ...", "code `let x = i32;` ...");
        t("type `Type<'static>` ...", "type `Type<'static>` ...");
        t("# top header", "top header");
        t("## header", "header");
    }

    #[test]
    fn test_markdown_html_escape() {
        fn t(input: &str, expect: &str) {
            let output = format!("{}", MarkdownHtml(input));
            assert_eq!(output, expect, "original: {}", input);
        }

        t("`Struct<'a, T>`", "<p><code>Struct&lt;&#39;a, T&gt;</code></p>");
        t("Struct<'a, T>", "<p>Struct&lt;&#39;a, T&gt;</p>");
    }
}
