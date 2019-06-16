//! Markdown formatting for rustdoc.
//!
//! This module implements markdown formatting through the pulldown-cmark
//! rust-library. This module exposes all of the
//! functionality through a unit struct, `Markdown`, which has an implementation
//! of `fmt::Display`. Example usage:
//!
//! ```
//! #![feature(rustc_private)]
//!
//! extern crate syntax;
//!
//! use syntax::edition::Edition;
//! use rustdoc::html::markdown::{IdMap, Markdown, ErrorCodes};
//! use std::cell::RefCell;
//!
//! let s = "My *markdown* _text_";
//! let mut id_map = IdMap::new();
//! let html = format!("{}", Markdown(s, &[], RefCell::new(&mut id_map),
//!                                   ErrorCodes::Yes, Edition::Edition2015));
//! // ... something using html
//! ```

#![allow(non_camel_case_types)]

use rustc_data_structures::fx::FxHashMap;
use std::cell::RefCell;
use std::collections::VecDeque;
use std::default::Default;
use std::fmt::{self, Write};
use std::borrow::Cow;
use std::ops::Range;
use std::str;
use syntax::edition::Edition;

use crate::html::toc::TocBuilder;
use crate::html::highlight;
use crate::test;

use pulldown_cmark::{html, CowStr, Event, Options, Parser, Tag};

fn opts() -> Options {
    Options::ENABLE_TABLES | Options::ENABLE_FOOTNOTES
}

/// A tuple struct that has the `fmt::Display` trait implemented.
/// When formatted, this struct will emit the HTML corresponding to the rendered
/// version of the contained markdown string.
pub struct Markdown<'a>(
    pub &'a str,
    /// A list of link replacements.
    pub &'a [(String, String)],
    /// The current list of used header IDs.
    pub RefCell<&'a mut IdMap>,
    /// Whether to allow the use of explicit error codes in doctest lang strings.
    pub ErrorCodes,
    /// Default edition to use when parsing doctests (to add a `fn main`).
    pub Edition,
);
/// A tuple struct like `Markdown` that renders the markdown with a table of contents.
pub struct MarkdownWithToc<'a>(
    pub &'a str,
    pub RefCell<&'a mut IdMap>,
    pub ErrorCodes,
    pub Edition,
);
/// A tuple struct like `Markdown` that renders the markdown escaping HTML tags.
pub struct MarkdownHtml<'a>(pub &'a str, pub RefCell<&'a mut IdMap>, pub ErrorCodes, pub Edition);
/// A tuple struct like `Markdown` that renders only the first paragraph.
pub struct MarkdownSummaryLine<'a>(pub &'a str, pub &'a [(String, String)]);

#[derive(Copy, Clone, PartialEq, Debug)]
pub enum ErrorCodes {
    Yes,
    No,
}

impl ErrorCodes {
    pub fn from(b: bool) -> Self {
        match b {
            true => ErrorCodes::Yes,
            false => ErrorCodes::No,
        }
    }

    pub fn as_bool(self) -> bool {
        match self {
            ErrorCodes::Yes => true,
            ErrorCodes::No => false,
        }
    }
}

/// Controls whether a line will be hidden or shown in HTML output.
///
/// All lines are used in documentation tests.
enum Line<'a> {
    Hidden(&'a str),
    Shown(Cow<'a, str>),
}

impl<'a> Line<'a> {
    fn for_html(self) -> Option<Cow<'a, str>> {
        match self {
            Line::Shown(l) => Some(l),
            Line::Hidden(_) => None,
        }
    }

    fn for_code(self) -> Cow<'a, str> {
        match self {
            Line::Shown(l) => l,
            Line::Hidden(l) => Cow::Borrowed(l),
        }
    }
}

// FIXME: There is a minor inconsistency here. For lines that start with ##, we
// have no easy way of removing a potential single space after the hashes, which
// is done in the single # case. This inconsistency seems okay, if non-ideal. In
// order to fix it we'd have to iterate to find the first non-# character, and
// then reallocate to remove it; which would make us return a String.
fn map_line(s: &str) -> Line<'_> {
    let trimmed = s.trim();
    if trimmed.starts_with("##") {
        Line::Shown(Cow::Owned(s.replacen("##", "#", 1)))
    } else if trimmed.starts_with("# ") {
        // # text
        Line::Hidden(&trimmed[2..])
    } else if trimmed == "#" {
        // We cannot handle '#text' because it could be #[attr].
        Line::Hidden("")
    } else {
        Line::Shown(Cow::Borrowed(s))
    }
}

/// Convert chars from a title for an id.
///
/// "Hello, world!" -> "hello-world"
fn slugify(c: char) -> Option<char> {
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
}

// Information about the playground if a URL has been specified, containing an
// optional crate name and the URL.
thread_local!(pub static PLAYGROUND: RefCell<Option<(Option<String>, String)>> = {
    RefCell::new(None)
});

/// Adds syntax highlighting and playground Run buttons to Rust code blocks.
struct CodeBlocks<'a, I: Iterator<Item = Event<'a>>> {
    inner: I,
    check_error_codes: ErrorCodes,
    edition: Edition,
}

impl<'a, I: Iterator<Item = Event<'a>>> CodeBlocks<'a, I> {
    fn new(iter: I, error_codes: ErrorCodes, edition: Edition) -> Self {
        CodeBlocks {
            inner: iter,
            check_error_codes: error_codes,
            edition,
        }
    }
}

impl<'a, I: Iterator<Item = Event<'a>>> Iterator for CodeBlocks<'a, I> {
    type Item = Event<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let event = self.inner.next();
        let compile_fail;
        let ignore;
        let edition;
        if let Some(Event::Start(Tag::CodeBlock(lang))) = event {
            let parse_result = LangString::parse(&lang, self.check_error_codes);
            if !parse_result.rust {
                return Some(Event::Start(Tag::CodeBlock(lang)));
            }
            compile_fail = parse_result.compile_fail;
            ignore = parse_result.ignore;
            edition = parse_result.edition;
        } else {
            return event;
        }

        let explicit_edition = edition.is_some();
        let edition = edition.unwrap_or(self.edition);

        let mut origtext = String::new();
        for event in &mut self.inner {
            match event {
                Event::End(Tag::CodeBlock(..)) => break,
                Event::Text(ref s) => {
                    origtext.push_str(s);
                }
                _ => {}
            }
        }
        let lines = origtext.lines().filter_map(|l| map_line(l).for_html());
        let text = lines.collect::<Vec<Cow<'_, str>>>().join("\n");
        PLAYGROUND.with(|play| {
            // insert newline to clearly separate it from the
            // previous block so we can shorten the html output
            let mut s = String::from("\n");
            let playground_button = play.borrow().as_ref().and_then(|&(ref krate, ref url)| {
                if url.is_empty() {
                    return None;
                }
                let test = origtext.lines()
                    .map(|l| map_line(l).for_code())
                    .collect::<Vec<Cow<'_, str>>>().join("\n");
                let krate = krate.as_ref().map(|s| &**s);
                let (test, _) = test::make_test(&test, krate, false,
                                           &Default::default(), edition);
                let channel = if test.contains("#![feature(") {
                    "&amp;version=nightly"
                } else {
                    ""
                };

                let edition_string = format!("&amp;edition={}", edition);

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
                    r#"<a class="test-arrow" target="_blank" href="{}?code={}{}{}">Run</a>"#,
                    url, test_escaped, channel, edition_string
                ))
            });

            let tooltip = if ignore {
                Some(("This example is not tested".to_owned(), "ignore"))
            } else if compile_fail {
                Some(("This example deliberately fails to compile".to_owned(), "compile_fail"))
            } else if explicit_edition {
                Some((format!("This code runs with edition {}", edition), "edition"))
            } else {
                None
            };

            if let Some((s1, s2)) = tooltip {
                s.push_str(&highlight::render_with_highlighting(
                    &text,
                    Some(&format!("rust-example-rendered{}",
                                  if ignore { " ignore" }
                                  else if compile_fail { " compile_fail" }
                                  else if explicit_edition { " edition " }
                                  else { "" })),
                    playground_button.as_ref().map(String::as_str),
                    Some((s1.as_str(), s2))));
                Some(Event::Html(s.into()))
            } else {
                s.push_str(&highlight::render_with_highlighting(
                    &text,
                    Some(&format!("rust-example-rendered{}",
                                  if ignore { " ignore" }
                                  else if compile_fail { " compile_fail" }
                                  else if explicit_edition { " edition " }
                                  else { "" })),
                    playground_button.as_ref().map(String::as_str),
                    None));
                Some(Event::Html(s.into()))
            }
        })
    }
}

/// Make headings links with anchor IDs and build up TOC.
struct LinkReplacer<'a, 'b, I: Iterator<Item = Event<'a>>> {
    inner: I,
    links: &'b [(String, String)],
}

impl<'a, 'b, I: Iterator<Item = Event<'a>>> LinkReplacer<'a, 'b, I> {
    fn new(iter: I, links: &'b [(String, String)]) -> Self {
        LinkReplacer {
            inner: iter,
            links,
        }
    }
}

impl<'a, 'b, I: Iterator<Item = Event<'a>>> Iterator for LinkReplacer<'a, 'b, I> {
    type Item = Event<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let event = self.inner.next();
        if let Some(Event::Start(Tag::Link(kind, dest, text))) = event {
            if let Some(&(_, ref replace)) = self.links.iter().find(|link| link.0 == *dest) {
                Some(Event::Start(Tag::Link(kind, replace.to_owned().into(), text)))
            } else {
                Some(Event::Start(Tag::Link(kind, dest, text)))
            }
        } else {
            event
        }
    }
}

/// Make headings links with anchor IDs and build up TOC.
struct HeadingLinks<'a, 'b, 'ids, I: Iterator<Item = Event<'a>>> {
    inner: I,
    toc: Option<&'b mut TocBuilder>,
    buf: VecDeque<Event<'a>>,
    id_map: &'ids mut IdMap,
}

impl<'a, 'b, 'ids, I: Iterator<Item = Event<'a>>> HeadingLinks<'a, 'b, 'ids, I> {
    fn new(iter: I, toc: Option<&'b mut TocBuilder>, ids: &'ids mut IdMap) -> Self {
        HeadingLinks {
            inner: iter,
            toc,
            buf: VecDeque::new(),
            id_map: ids,
        }
    }
}

impl<'a, 'b, 'ids, I: Iterator<Item = Event<'a>>> Iterator for HeadingLinks<'a, 'b, 'ids, I> {
    type Item = Event<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(e) = self.buf.pop_front() {
            return Some(e);
        }

        let event = self.inner.next();
        if let Some(Event::Start(Tag::Header(level))) = event {
            let mut id = String::new();
            for event in &mut self.inner {
                match &event {
                    Event::End(Tag::Header(..)) => break,
                    Event::Text(text) | Event::Code(text) => {
                        id.extend(text.chars().filter_map(slugify));
                    }
                    _ => {},
                }
                self.buf.push_back(event);
            }
            let id = self.id_map.derive(id);

            if let Some(ref mut builder) = self.toc {
                let mut html_header = String::new();
                html::push_html(&mut html_header, self.buf.iter().cloned());
                let sec = builder.push(level as u32, html_header, id.clone());
                self.buf.push_front(Event::InlineHtml(format!("{} ", sec).into()));
            }

            self.buf.push_back(Event::InlineHtml(format!("</a></h{}>", level).into()));

            let start_tags = format!("<h{level} id=\"{id}\" class=\"section-header\">\
                                      <a href=\"#{id}\">",
                                     id = id,
                                     level = level);
            return Some(Event::InlineHtml(start_tags.into()));
        }
        event
    }
}

/// Extracts just the first paragraph.
struct SummaryLine<'a, I: Iterator<Item = Event<'a>>> {
    inner: I,
    started: bool,
    depth: u32,
}

impl<'a, I: Iterator<Item = Event<'a>>> SummaryLine<'a, I> {
    fn new(iter: I) -> Self {
        SummaryLine {
            inner: iter,
            started: false,
            depth: 0,
        }
    }
}

fn check_if_allowed_tag(t: &Tag<'_>) -> bool {
    match *t {
        Tag::Paragraph
        | Tag::Item
        | Tag::Emphasis
        | Tag::Strong
        | Tag::Link(..)
        | Tag::BlockQuote => true,
        _ => false,
    }
}

impl<'a, I: Iterator<Item = Event<'a>>> Iterator for SummaryLine<'a, I> {
    type Item = Event<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.started && self.depth == 0 {
            return None;
        }
        if !self.started {
            self.started = true;
        }
        while let Some(event) = self.inner.next() {
            let mut is_start = true;
            let is_allowed_tag = match event {
                Event::Start(Tag::CodeBlock(_)) | Event::End(Tag::CodeBlock(_)) => {
                    return None;
                }
                Event::Start(ref c) => {
                    self.depth += 1;
                    check_if_allowed_tag(c)
                }
                Event::End(ref c) => {
                    self.depth -= 1;
                    is_start = false;
                    check_if_allowed_tag(c)
                }
                _ => {
                    true
                }
            };
            return if is_allowed_tag == false {
                if is_start {
                    Some(Event::Start(Tag::Paragraph))
                } else {
                    Some(Event::End(Tag::Paragraph))
                }
            } else {
                Some(event)
            };
        }
        None
    }
}

/// Moves all footnote definitions to the end and add back links to the
/// references.
struct Footnotes<'a, I: Iterator<Item = Event<'a>>> {
    inner: I,
    footnotes: FxHashMap<String, (Vec<Event<'a>>, u16)>,
}

impl<'a, I: Iterator<Item = Event<'a>>> Footnotes<'a, I> {
    fn new(iter: I) -> Self {
        Footnotes {
            inner: iter,
            footnotes: FxHashMap::default(),
        }
    }
    fn get_entry(&mut self, key: &str) -> &mut (Vec<Event<'a>>, u16) {
        let new_id = self.footnotes.keys().count() + 1;
        let key = key.to_owned();
        self.footnotes.entry(key).or_insert((Vec::new(), new_id as u16))
    }
}

impl<'a, I: Iterator<Item = Event<'a>>> Iterator for Footnotes<'a, I> {
    type Item = Event<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.inner.next() {
                Some(Event::FootnoteReference(ref reference)) => {
                    let entry = self.get_entry(&reference);
                    let reference = format!("<sup id=\"fnref{0}\"><a href=\"#fn{0}\">{0}\
                                             </a></sup>",
                                            (*entry).1);
                    return Some(Event::Html(reference.into()));
                }
                Some(Event::Start(Tag::FootnoteDefinition(def))) => {
                    let mut content = Vec::new();
                    for event in &mut self.inner {
                        if let Event::End(Tag::FootnoteDefinition(..)) = event {
                            break;
                        }
                        content.push(event);
                    }
                    let entry = self.get_entry(&def);
                    (*entry).0 = content;
                }
                Some(e) => return Some(e),
                None => {
                    if !self.footnotes.is_empty() {
                        let mut v: Vec<_> = self.footnotes.drain().map(|(_, x)| x).collect();
                        v.sort_by(|a, b| a.1.cmp(&b.1));
                        let mut ret = String::from("<div class=\"footnotes\"><hr><ol>");
                        for (mut content, id) in v {
                            write!(ret, "<li id=\"fn{}\">", id).unwrap();
                            let mut is_paragraph = false;
                            if let Some(&Event::End(Tag::Paragraph)) = content.last() {
                                content.pop();
                                is_paragraph = true;
                            }
                            html::push_html(&mut ret, content.into_iter());
                            write!(ret,
                                   "&nbsp;<a href=\"#fnref{}\" rev=\"footnote\">↩</a>",
                                   id).unwrap();
                            if is_paragraph {
                                ret.push_str("</p>");
                            }
                            ret.push_str("</li>");
                        }
                        ret.push_str("</ol></div>");
                        return Some(Event::Html(ret.into()));
                    } else {
                        return None;
                    }
                }
            }
        }
    }
}

pub fn find_testable_code<T: test::Tester>(doc: &str, tests: &mut T, error_codes: ErrorCodes) {
    let mut parser = Parser::new(doc);
    let mut prev_offset = 0;
    let mut nb_lines = 0;
    let mut register_header = None;
    while let Some(event) = parser.next() {
        match event {
            Event::Start(Tag::CodeBlock(s)) => {
                let offset = parser.get_offset();

                let block_info = if s.is_empty() {
                    LangString::all_false()
                } else {
                    LangString::parse(&*s, error_codes)
                };
                if !block_info.rust {
                    continue;
                }
                let mut test_s = String::new();

                while let Some(Event::Text(s)) = parser.next() {
                    test_s.push_str(&s);
                }

                let text = test_s
                    .lines()
                    .map(|l| map_line(l).for_code())
                    .collect::<Vec<Cow<'_, str>>>()
                    .join("\n");
                nb_lines += doc[prev_offset..offset].lines().count();
                let line = tests.get_line() + nb_lines;
                tests.add_test(text, block_info, line);
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
pub struct LangString {
    original: String,
    pub should_panic: bool,
    pub no_run: bool,
    pub ignore: bool,
    pub rust: bool,
    pub test_harness: bool,
    pub compile_fail: bool,
    pub error_codes: Vec<String>,
    pub allow_fail: bool,
    pub edition: Option<Edition>
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
            allow_fail: false,
            edition: None,
        }
    }

    fn parse(string: &str, allow_error_code_check: ErrorCodes) -> LangString {
        let allow_error_code_check = allow_error_code_check.as_bool();
        let mut seen_rust_tags = false;
        let mut seen_other_tags = false;
        let mut data = LangString::all_false();

        data.original = string.to_owned();
        let tokens = string.split(|c: char|
            !(c == '_' || c == '-' || c.is_alphanumeric())
        );

        for token in tokens {
            match token.trim() {
                "" => {},
                "should_panic" => {
                    data.should_panic = true;
                    seen_rust_tags = seen_other_tags == false;
                }
                "no_run" => { data.no_run = true; seen_rust_tags = !seen_other_tags; }
                "ignore" => { data.ignore = true; seen_rust_tags = !seen_other_tags; }
                "allow_fail" => { data.allow_fail = true; seen_rust_tags = !seen_other_tags; }
                "rust" => { data.rust = true; seen_rust_tags = true; }
                "test_harness" => {
                    data.test_harness = true;
                    seen_rust_tags = !seen_other_tags || seen_rust_tags;
                }
                "compile_fail" => {
                    data.compile_fail = true;
                    seen_rust_tags = !seen_other_tags || seen_rust_tags;
                    data.no_run = true;
                }
                x if allow_error_code_check && x.starts_with("edition") => {
                    // allow_error_code_check is true if we're on nightly, which
                    // is needed for edition support
                    data.edition = x[7..].parse::<Edition>().ok();
                }
                x if allow_error_code_check && x.starts_with("E") && x.len() == 5 => {
                    if x[1..].parse::<u32>().is_ok() {
                        data.error_codes.push(x.to_owned());
                        seen_rust_tags = !seen_other_tags || seen_rust_tags;
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
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Markdown(md, links, ref ids, codes, edition) = *self;
        let mut ids = ids.borrow_mut();

        // This is actually common enough to special-case
        if md.is_empty() { return Ok(()) }
        let replacer = |_: &str, s: &str| {
            if let Some(&(_, ref replace)) = links.into_iter().find(|link| &*link.0 == s) {
                Some((replace.clone(), s.to_owned()))
            } else {
                None
            }
        };

        let p = Parser::new_with_broken_link_callback(md, opts(), Some(&replacer));

        let mut s = String::with_capacity(md.len() * 3 / 2);

        let p = HeadingLinks::new(p, None, &mut ids);
        let p = LinkReplacer::new(p, links);
        let p = CodeBlocks::new(p, codes, edition);
        let p = Footnotes::new(p);
        html::push_html(&mut s, p);

        fmt.write_str(&s)
    }
}

impl<'a> fmt::Display for MarkdownWithToc<'a> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        let MarkdownWithToc(md, ref ids, codes, edition) = *self;
        let mut ids = ids.borrow_mut();

        let p = Parser::new_ext(md, opts());

        let mut s = String::with_capacity(md.len() * 3 / 2);

        let mut toc = TocBuilder::new();

        {
            let p = HeadingLinks::new(p, Some(&mut toc), &mut ids);
            let p = CodeBlocks::new(p, codes, edition);
            let p = Footnotes::new(p);
            html::push_html(&mut s, p);
        }

        write!(fmt, "<nav id=\"TOC\">{}</nav>", toc.into_toc())?;

        fmt.write_str(&s)
    }
}

impl<'a> fmt::Display for MarkdownHtml<'a> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        let MarkdownHtml(md, ref ids, codes, edition) = *self;
        let mut ids = ids.borrow_mut();

        // This is actually common enough to special-case
        if md.is_empty() { return Ok(()) }
        let p = Parser::new_ext(md, opts());

        // Treat inline HTML as plain text.
        let p = p.map(|event| match event {
            Event::Html(text) | Event::InlineHtml(text) => Event::Text(text),
            _ => event
        });

        let mut s = String::with_capacity(md.len() * 3 / 2);

        let p = HeadingLinks::new(p, None, &mut ids);
        let p = CodeBlocks::new(p, codes, edition);
        let p = Footnotes::new(p);
        html::push_html(&mut s, p);

        fmt.write_str(&s)
    }
}

impl<'a> fmt::Display for MarkdownSummaryLine<'a> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        let MarkdownSummaryLine(md, links) = *self;
        // This is actually common enough to special-case
        if md.is_empty() { return Ok(()) }

        let replacer = |_: &str, s: &str| {
            if let Some(&(_, ref replace)) = links.into_iter().find(|link| &*link.0 == s) {
                Some((replace.clone(), s.to_owned()))
            } else {
                None
            }
        };

        let p = Parser::new_with_broken_link_callback(md, Options::empty(), Some(&replacer));

        let mut s = String::new();

        html::push_html(&mut s, LinkReplacer::new(SummaryLine::new(p), links));

        fmt.write_str(&s)
    }
}

pub fn plain_summary_line(md: &str) -> String {
    plain_summary_line_full(md, false)
}

pub fn plain_summary_line_full(md: &str, limit_length: bool) -> String {
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
                Event::Start(Tag::Header(_)) => (None, 1),
                Event::Code(code) => (Some(format!("`{}`", code)), 0),
                Event::Text(ref s) if self.is_in > 0 => (Some(s.as_ref().to_owned()), 0),
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
    if limit_length && s.chars().count() > 60 {
        let mut len = 0;
        let mut ret = s.split_whitespace()
                       .take_while(|p| {
                           // + 1 for the added character after the word.
                           len += p.chars().count() + 1;
                           len < 60
                       })
                       .collect::<Vec<_>>()
                       .join(" ");
        ret.push('…');
        ret
    } else {
        s
    }
}

pub fn markdown_links(md: &str) -> Vec<(String, Option<Range<usize>>)> {
    if md.is_empty() {
        return vec![];
    }

    let mut links = vec![];
    let shortcut_links = RefCell::new(vec![]);

    {
        let locate = |s: &str| unsafe {
            let s_start = s.as_ptr();
            let s_end = s_start.add(s.len());
            let md_start = md.as_ptr();
            let md_end = md_start.add(md.len());
            if md_start <= s_start && s_end <= md_end {
                let start = s_start.offset_from(md_start) as usize;
                let end = s_end.offset_from(md_start) as usize;
                Some(start..end)
            } else {
                None
            }
        };

        let push = |_: &str, s: &str| {
            shortcut_links.borrow_mut().push((s.to_owned(), locate(s)));
            None
        };
        let p = Parser::new_with_broken_link_callback(md, opts(), Some(&push));

        // There's no need to thread an IdMap through to here because
        // the IDs generated aren't going to be emitted anywhere.
        let mut ids = IdMap::new();
        let iter = Footnotes::new(HeadingLinks::new(p, None, &mut ids));

        for ev in iter {
            if let Event::Start(Tag::Link(_, dest, _)) = ev {
                debug!("found link: {}", dest);
                links.push(match dest {
                    CowStr::Borrowed(s) => (s.to_owned(), locate(s)),
                    s @ CowStr::Boxed(..) | s @ CowStr::Inlined(..) => (s.into_string(), None),
                });
            }
        }
    }

    let mut shortcut_links = shortcut_links.into_inner();
    links.extend(shortcut_links.drain(..));

    links
}

#[derive(Debug)]
crate struct RustCodeBlock {
    /// The range in the markdown that the code block occupies. Note that this includes the fences
    /// for fenced code blocks.
    pub range: Range<usize>,
    /// The range in the markdown that the code within the code block occupies.
    pub code: Range<usize>,
    pub is_fenced: bool,
    pub syntax: Option<String>,
}

/// Returns a range of bytes for each code block in the markdown that is tagged as `rust` or
/// untagged (and assumed to be rust).
crate fn rust_code_blocks(md: &str) -> Vec<RustCodeBlock> {
    let mut code_blocks = vec![];

    if md.is_empty() {
        return code_blocks;
    }

    let mut p = Parser::new_ext(md, opts());

    let mut code_block_start = 0;
    let mut code_start = 0;
    let mut is_fenced = false;
    let mut previous_offset = 0;
    let mut in_rust_code_block = false;
    while let Some(event) = p.next() {
        let offset = p.get_offset();

        match event {
            Event::Start(Tag::CodeBlock(syntax)) => {
                let lang_string = if syntax.is_empty() {
                    LangString::all_false()
                } else {
                    LangString::parse(&*syntax, ErrorCodes::Yes)
                };

                if lang_string.rust {
                    in_rust_code_block = true;

                    code_start = offset;
                    code_block_start = match md[previous_offset..offset].find("```") {
                        Some(fence_idx) => {
                            is_fenced = true;
                            previous_offset + fence_idx
                        }
                        None => offset,
                    };
                }
            }
            Event::End(Tag::CodeBlock(syntax)) if in_rust_code_block => {
                in_rust_code_block = false;

                let code_block_end = if is_fenced {
                    let fence_str = &md[previous_offset..offset]
                        .chars()
                        .rev()
                        .collect::<String>();
                    fence_str
                        .find("```")
                        .map(|fence_idx| offset - fence_idx)
                        .unwrap_or_else(|| offset)
                } else if md
                    .as_bytes()
                    .get(offset)
                    .map(|b| *b == b'\n')
                    .unwrap_or_default()
                {
                    offset - 1
                } else {
                    offset
                };

                let code_end = if is_fenced {
                    previous_offset
                } else {
                    code_block_end
                };

                code_blocks.push(RustCodeBlock {
                    is_fenced,
                    range: Range {
                        start: code_block_start,
                        end: code_block_end,
                    },
                    code: Range {
                        start: code_start,
                        end: code_end,
                    },
                    syntax: if !syntax.is_empty() {
                        Some(syntax.into_string())
                    } else {
                        None
                    },
                });
            }
            _ => (),
        }

        previous_offset = offset;
    }

    code_blocks
}

#[derive(Clone, Default, Debug)]
pub struct IdMap {
    map: FxHashMap<String, usize>,
}

impl IdMap {
    pub fn new() -> Self {
        IdMap::default()
    }

    pub fn populate<I: IntoIterator<Item=String>>(&mut self, ids: I) {
        for id in ids {
            let _ = self.derive(id);
        }
    }

    pub fn reset(&mut self) {
        self.map = FxHashMap::default();
    }

    pub fn derive(&mut self, candidate: String) -> String {
        let id = match self.map.get_mut(&candidate) {
            None => candidate,
            Some(a) => {
                let id = format!("{}-{}", candidate, *a);
                *a += 1;
                id
            }
        };

        self.map.insert(id.clone(), 1);
        id
    }
}

#[cfg(test)]
#[test]
fn test_unique_id() {
    let input = ["foo", "examples", "examples", "method.into_iter","examples",
                 "method.into_iter", "foo", "main", "search", "methods",
                 "examples", "method.into_iter", "assoc_type.Item", "assoc_type.Item"];
    let expected = ["foo", "examples", "examples-1", "method.into_iter", "examples-2",
                    "method.into_iter-1", "foo-1", "main", "search", "methods",
                    "examples-3", "method.into_iter-2", "assoc_type.Item", "assoc_type.Item-1"];

    let map = RefCell::new(IdMap::new());
    let test = || {
        let mut map = map.borrow_mut();
        let actual: Vec<String> = input.iter().map(|s| map.derive(s.to_string())).collect();
        assert_eq!(&actual[..], expected);
    };
    test();
    map.borrow_mut().reset();
    test();
}

#[cfg(test)]
mod tests;
