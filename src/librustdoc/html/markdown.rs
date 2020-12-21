//! Markdown formatting for rustdoc.
//!
//! This module implements markdown formatting through the pulldown-cmark library.
//!
//! ```
//! #![feature(rustc_private)]
//!
//! extern crate rustc_span;
//!
//! use rustc_span::edition::Edition;
//! use rustdoc::html::markdown::{IdMap, Markdown, ErrorCodes};
//!
//! let s = "My *markdown* _text_";
//! let mut id_map = IdMap::new();
//! let md = Markdown(s, &[], &mut id_map, ErrorCodes::Yes, Edition::Edition2015, &None);
//! let html = md.into_string();
//! // ... something using html
//! ```

use rustc_data_structures::fx::FxHashMap;
use rustc_hir::def_id::DefId;
use rustc_hir::HirId;
use rustc_middle::ty::TyCtxt;
use rustc_session::lint;
use rustc_span::edition::Edition;
use rustc_span::Span;
use std::borrow::Cow;
use std::collections::VecDeque;
use std::default::Default;
use std::fmt::Write;
use std::ops::Range;
use std::str;

use crate::clean::RenderedLink;
use crate::doctest;
use crate::html::highlight;
use crate::html::toc::TocBuilder;

use pulldown_cmark::{html, BrokenLink, CodeBlockKind, CowStr, Event, Options, Parser, Tag};

#[cfg(test)]
mod tests;

/// Options for rendering Markdown in the main body of documentation.
pub(crate) fn opts() -> Options {
    Options::ENABLE_TABLES | Options::ENABLE_FOOTNOTES | Options::ENABLE_STRIKETHROUGH
}

/// A subset of [`opts()`] used for rendering summaries.
pub(crate) fn summary_opts() -> Options {
    Options::ENABLE_STRIKETHROUGH
}

/// When `to_string` is called, this struct will emit the HTML corresponding to
/// the rendered version of the contained markdown string.
pub struct Markdown<'a>(
    pub &'a str,
    /// A list of link replacements.
    pub &'a [RenderedLink],
    /// The current list of used header IDs.
    pub &'a mut IdMap,
    /// Whether to allow the use of explicit error codes in doctest lang strings.
    pub ErrorCodes,
    /// Default edition to use when parsing doctests (to add a `fn main`).
    pub Edition,
    pub &'a Option<Playground>,
);
/// A tuple struct like `Markdown` that renders the markdown with a table of contents.
crate struct MarkdownWithToc<'a>(
    crate &'a str,
    crate &'a mut IdMap,
    crate ErrorCodes,
    crate Edition,
    crate &'a Option<Playground>,
);
/// A tuple struct like `Markdown` that renders the markdown escaping HTML tags.
crate struct MarkdownHtml<'a>(
    crate &'a str,
    crate &'a mut IdMap,
    crate ErrorCodes,
    crate Edition,
    crate &'a Option<Playground>,
);
/// A tuple struct like `Markdown` that renders only the first paragraph.
crate struct MarkdownSummaryLine<'a>(pub &'a str, pub &'a [RenderedLink]);

#[derive(Copy, Clone, PartialEq, Debug)]
pub enum ErrorCodes {
    Yes,
    No,
}

impl ErrorCodes {
    crate fn from(b: bool) -> Self {
        match b {
            true => ErrorCodes::Yes,
            false => ErrorCodes::No,
        }
    }

    crate fn as_bool(self) -> bool {
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
    } else if let Some(stripped) = trimmed.strip_prefix("# ") {
        // # text
        Line::Hidden(&stripped)
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
        if c.is_ascii() { Some(c.to_ascii_lowercase()) } else { Some(c) }
    } else if c.is_whitespace() && c.is_ascii() {
        Some('-')
    } else {
        None
    }
}

#[derive(Clone, Debug)]
pub struct Playground {
    pub crate_name: Option<String>,
    pub url: String,
}

/// Adds syntax highlighting and playground Run buttons to Rust code blocks.
struct CodeBlocks<'p, 'a, I: Iterator<Item = Event<'a>>> {
    inner: I,
    check_error_codes: ErrorCodes,
    edition: Edition,
    // Information about the playground if a URL has been specified, containing an
    // optional crate name and the URL.
    playground: &'p Option<Playground>,
}

impl<'p, 'a, I: Iterator<Item = Event<'a>>> CodeBlocks<'p, 'a, I> {
    fn new(
        iter: I,
        error_codes: ErrorCodes,
        edition: Edition,
        playground: &'p Option<Playground>,
    ) -> Self {
        CodeBlocks { inner: iter, check_error_codes: error_codes, edition, playground }
    }
}

impl<'a, I: Iterator<Item = Event<'a>>> Iterator for CodeBlocks<'_, 'a, I> {
    type Item = Event<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let event = self.inner.next();
        let compile_fail;
        let should_panic;
        let ignore;
        let edition;
        if let Some(Event::Start(Tag::CodeBlock(kind))) = event {
            let parse_result = match kind {
                CodeBlockKind::Fenced(ref lang) => {
                    LangString::parse_without_check(&lang, self.check_error_codes, false)
                }
                CodeBlockKind::Indented => Default::default(),
            };
            if !parse_result.rust {
                return Some(Event::Start(Tag::CodeBlock(kind)));
            }
            compile_fail = parse_result.compile_fail;
            should_panic = parse_result.should_panic;
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
        // insert newline to clearly separate it from the
        // previous block so we can shorten the html output
        let mut s = String::from("\n");
        let playground_button = self.playground.as_ref().and_then(|playground| {
            let krate = &playground.crate_name;
            let url = &playground.url;
            if url.is_empty() {
                return None;
            }
            let test = origtext
                .lines()
                .map(|l| map_line(l).for_code())
                .collect::<Vec<Cow<'_, str>>>()
                .join("\n");
            let krate = krate.as_ref().map(|s| &**s);
            let (test, _, _) =
                doctest::make_test(&test, krate, false, &Default::default(), edition);
            let channel = if test.contains("#![feature(") { "&amp;version=nightly" } else { "" };

            let edition_string = format!("&amp;edition={}", edition);

            // These characters don't need to be escaped in a URI.
            // FIXME: use a library function for percent encoding.
            fn dont_escape(c: u8) -> bool {
                (b'a' <= c && c <= b'z')
                    || (b'A' <= c && c <= b'Z')
                    || (b'0' <= c && c <= b'9')
                    || c == b'-'
                    || c == b'_'
                    || c == b'.'
                    || c == b'~'
                    || c == b'!'
                    || c == b'\''
                    || c == b'('
                    || c == b')'
                    || c == b'*'
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

        let tooltip = if ignore != Ignore::None {
            Some(("This example is not tested".to_owned(), "ignore"))
        } else if compile_fail {
            Some(("This example deliberately fails to compile".to_owned(), "compile_fail"))
        } else if should_panic {
            Some(("This example panics".to_owned(), "should_panic"))
        } else if explicit_edition {
            Some((format!("This code runs with edition {}", edition), "edition"))
        } else {
            None
        };

        if let Some((s1, s2)) = tooltip {
            s.push_str(&highlight::render_with_highlighting(
                text,
                Some(&format!(
                    "rust-example-rendered{}",
                    if ignore != Ignore::None {
                        " ignore"
                    } else if compile_fail {
                        " compile_fail"
                    } else if should_panic {
                        " should_panic"
                    } else if explicit_edition {
                        " edition "
                    } else {
                        ""
                    }
                )),
                playground_button.as_deref(),
                Some((s1.as_str(), s2)),
            ));
            Some(Event::Html(s.into()))
        } else {
            s.push_str(&highlight::render_with_highlighting(
                text,
                Some(&format!(
                    "rust-example-rendered{}",
                    if ignore != Ignore::None {
                        " ignore"
                    } else if compile_fail {
                        " compile_fail"
                    } else if should_panic {
                        " should_panic"
                    } else if explicit_edition {
                        " edition "
                    } else {
                        ""
                    }
                )),
                playground_button.as_deref(),
                None,
            ));
            Some(Event::Html(s.into()))
        }
    }
}

/// Make headings links with anchor IDs and build up TOC.
struct LinkReplacer<'a, I: Iterator<Item = Event<'a>>> {
    inner: I,
    links: &'a [RenderedLink],
    shortcut_link: Option<&'a RenderedLink>,
}

impl<'a, I: Iterator<Item = Event<'a>>> LinkReplacer<'a, I> {
    fn new(iter: I, links: &'a [RenderedLink]) -> Self {
        LinkReplacer { inner: iter, links, shortcut_link: None }
    }
}

impl<'a, I: Iterator<Item = Event<'a>>> Iterator for LinkReplacer<'a, I> {
    type Item = Event<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        use pulldown_cmark::LinkType;

        let mut event = self.inner.next();

        // Replace intra-doc links and remove disambiguators from shortcut links (`[fn@f]`).
        match &mut event {
            // This is a shortcut link that was resolved by the broken_link_callback: `[fn@f]`
            // Remove any disambiguator.
            Some(Event::Start(Tag::Link(
                // [fn@f] or [fn@f][]
                LinkType::ShortcutUnknown | LinkType::CollapsedUnknown,
                dest,
                title,
            ))) => {
                debug!("saw start of shortcut link to {} with title {}", dest, title);
                // If this is a shortcut link, it was resolved by the broken_link_callback.
                // So the URL will already be updated properly.
                let link = self.links.iter().find(|&link| *link.href == **dest);
                // Since this is an external iterator, we can't replace the inner text just yet.
                // Store that we saw a link so we know to replace it later.
                if let Some(link) = link {
                    trace!("it matched");
                    assert!(self.shortcut_link.is_none(), "shortcut links cannot be nested");
                    self.shortcut_link = Some(link);
                }
            }
            // Now that we're done with the shortcut link, don't replace any more text.
            Some(Event::End(Tag::Link(
                LinkType::ShortcutUnknown | LinkType::CollapsedUnknown,
                dest,
                _,
            ))) => {
                debug!("saw end of shortcut link to {}", dest);
                if self.links.iter().any(|link| *link.href == **dest) {
                    assert!(self.shortcut_link.is_some(), "saw closing link without opening tag");
                    self.shortcut_link = None;
                }
            }
            // Handle backticks in inline code blocks, but only if we're in the middle of a shortcut link.
            // [`fn@f`]
            Some(Event::Code(text)) => {
                trace!("saw code {}", text);
                if let Some(link) = self.shortcut_link {
                    trace!("original text was {}", link.original_text);
                    // NOTE: this only replaces if the code block is the *entire* text.
                    // If only part of the link has code highlighting, the disambiguator will not be removed.
                    // e.g. [fn@`f`]
                    // This is a limitation from `collect_intra_doc_links`: it passes a full link,
                    // and does not distinguish at all between code blocks.
                    // So we could never be sure we weren't replacing too much:
                    // [fn@my_`f`unc] is treated the same as [my_func()] in that pass.
                    //
                    // NOTE: &[1..len() - 1] is to strip the backticks
                    if **text == link.original_text[1..link.original_text.len() - 1] {
                        debug!("replacing {} with {}", text, link.new_text);
                        *text = CowStr::Borrowed(&link.new_text);
                    }
                }
            }
            // Replace plain text in links, but only in the middle of a shortcut link.
            // [fn@f]
            Some(Event::Text(text)) => {
                trace!("saw text {}", text);
                if let Some(link) = self.shortcut_link {
                    trace!("original text was {}", link.original_text);
                    // NOTE: same limitations as `Event::Code`
                    if **text == *link.original_text {
                        debug!("replacing {} with {}", text, link.new_text);
                        *text = CowStr::Borrowed(&link.new_text);
                    }
                }
            }
            // If this is a link, but not a shortcut link,
            // replace the URL, since the broken_link_callback was not called.
            Some(Event::Start(Tag::Link(_, dest, _))) => {
                if let Some(link) = self.links.iter().find(|&link| *link.original_text == **dest) {
                    *dest = CowStr::Borrowed(link.href.as_ref());
                }
            }
            // Anything else couldn't have been a valid Rust path, so no need to replace the text.
            _ => {}
        }

        // Yield the modified event
        event
    }
}

/// Make headings links with anchor IDs and build up TOC.
struct HeadingLinks<'a, 'b, 'ids, I> {
    inner: I,
    toc: Option<&'b mut TocBuilder>,
    buf: VecDeque<(Event<'a>, Range<usize>)>,
    id_map: &'ids mut IdMap,
}

impl<'a, 'b, 'ids, I> HeadingLinks<'a, 'b, 'ids, I> {
    fn new(iter: I, toc: Option<&'b mut TocBuilder>, ids: &'ids mut IdMap) -> Self {
        HeadingLinks { inner: iter, toc, buf: VecDeque::new(), id_map: ids }
    }
}

impl<'a, 'b, 'ids, I: Iterator<Item = (Event<'a>, Range<usize>)>> Iterator
    for HeadingLinks<'a, 'b, 'ids, I>
{
    type Item = (Event<'a>, Range<usize>);

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(e) = self.buf.pop_front() {
            return Some(e);
        }

        let event = self.inner.next();
        if let Some((Event::Start(Tag::Heading(level)), _)) = event {
            let mut id = String::new();
            for event in &mut self.inner {
                match &event.0 {
                    Event::End(Tag::Heading(..)) => break,
                    Event::Start(Tag::Link(_, _, _)) | Event::End(Tag::Link(..)) => {}
                    Event::Text(text) | Event::Code(text) => {
                        id.extend(text.chars().filter_map(slugify));
                        self.buf.push_back(event);
                    }
                    _ => self.buf.push_back(event),
                }
            }
            let id = self.id_map.derive(id);

            if let Some(ref mut builder) = self.toc {
                let mut html_header = String::new();
                html::push_html(&mut html_header, self.buf.iter().map(|(ev, _)| ev.clone()));
                let sec = builder.push(level as u32, html_header, id.clone());
                self.buf.push_front((Event::Html(format!("{} ", sec).into()), 0..0));
            }

            self.buf.push_back((Event::Html(format!("</a></h{}>", level).into()), 0..0));

            let start_tags = format!(
                "<h{level} id=\"{id}\" class=\"section-header\">\
                    <a href=\"#{id}\">",
                id = id,
                level = level
            );
            return Some((Event::Html(start_tags.into()), 0..0));
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
        SummaryLine { inner: iter, started: false, depth: 0 }
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
        if let Some(event) = self.inner.next() {
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
                _ => true,
            };
            return if !is_allowed_tag {
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
struct Footnotes<'a, I> {
    inner: I,
    footnotes: FxHashMap<String, (Vec<Event<'a>>, u16)>,
}

impl<'a, I> Footnotes<'a, I> {
    fn new(iter: I) -> Self {
        Footnotes { inner: iter, footnotes: FxHashMap::default() }
    }

    fn get_entry(&mut self, key: &str) -> &mut (Vec<Event<'a>>, u16) {
        let new_id = self.footnotes.keys().count() + 1;
        let key = key.to_owned();
        self.footnotes.entry(key).or_insert((Vec::new(), new_id as u16))
    }
}

impl<'a, I: Iterator<Item = (Event<'a>, Range<usize>)>> Iterator for Footnotes<'a, I> {
    type Item = (Event<'a>, Range<usize>);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.inner.next() {
                Some((Event::FootnoteReference(ref reference), range)) => {
                    let entry = self.get_entry(&reference);
                    let reference = format!(
                        "<sup id=\"fnref{0}\"><a href=\"#fn{0}\">{0}</a></sup>",
                        (*entry).1
                    );
                    return Some((Event::Html(reference.into()), range));
                }
                Some((Event::Start(Tag::FootnoteDefinition(def)), _)) => {
                    let mut content = Vec::new();
                    for (event, _) in &mut self.inner {
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
                            write!(ret, "&nbsp;<a href=\"#fnref{}\" rev=\"footnote\">↩</a>", id)
                                .unwrap();
                            if is_paragraph {
                                ret.push_str("</p>");
                            }
                            ret.push_str("</li>");
                        }
                        ret.push_str("</ol></div>");
                        return Some((Event::Html(ret.into()), 0..0));
                    } else {
                        return None;
                    }
                }
            }
        }
    }
}

crate fn find_testable_code<T: doctest::Tester>(
    doc: &str,
    tests: &mut T,
    error_codes: ErrorCodes,
    enable_per_target_ignores: bool,
    extra_info: Option<&ExtraInfo<'_, '_>>,
) {
    let mut parser = Parser::new(doc).into_offset_iter();
    let mut prev_offset = 0;
    let mut nb_lines = 0;
    let mut register_header = None;
    while let Some((event, offset)) = parser.next() {
        match event {
            Event::Start(Tag::CodeBlock(kind)) => {
                let block_info = match kind {
                    CodeBlockKind::Fenced(ref lang) => {
                        if lang.is_empty() {
                            Default::default()
                        } else {
                            LangString::parse(
                                lang,
                                error_codes,
                                enable_per_target_ignores,
                                extra_info,
                            )
                        }
                    }
                    CodeBlockKind::Indented => Default::default(),
                };
                if !block_info.rust {
                    continue;
                }

                let mut test_s = String::new();

                while let Some((Event::Text(s), _)) = parser.next() {
                    test_s.push_str(&s);
                }
                let text = test_s
                    .lines()
                    .map(|l| map_line(l).for_code())
                    .collect::<Vec<Cow<'_, str>>>()
                    .join("\n");

                nb_lines += doc[prev_offset..offset.start].lines().count();
                let line = tests.get_line() + nb_lines + 1;
                tests.add_test(text, block_info, line);
                prev_offset = offset.start;
            }
            Event::Start(Tag::Heading(level)) => {
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

crate struct ExtraInfo<'a, 'b> {
    hir_id: Option<HirId>,
    item_did: Option<DefId>,
    sp: Span,
    tcx: &'a TyCtxt<'b>,
}

impl<'a, 'b> ExtraInfo<'a, 'b> {
    crate fn new(tcx: &'a TyCtxt<'b>, hir_id: HirId, sp: Span) -> ExtraInfo<'a, 'b> {
        ExtraInfo { hir_id: Some(hir_id), item_did: None, sp, tcx }
    }

    crate fn new_did(tcx: &'a TyCtxt<'b>, did: DefId, sp: Span) -> ExtraInfo<'a, 'b> {
        ExtraInfo { hir_id: None, item_did: Some(did), sp, tcx }
    }

    fn error_invalid_codeblock_attr(&self, msg: &str, help: &str) {
        let hir_id = match (self.hir_id, self.item_did) {
            (Some(h), _) => h,
            (None, Some(item_did)) => {
                match item_did.as_local() {
                    Some(item_did) => self.tcx.hir().local_def_id_to_hir_id(item_did),
                    None => {
                        // If non-local, no need to check anything.
                        return;
                    }
                }
            }
            (None, None) => return,
        };
        self.tcx.struct_span_lint_hir(
            lint::builtin::INVALID_CODEBLOCK_ATTRIBUTES,
            hir_id,
            self.sp,
            |lint| {
                let mut diag = lint.build(msg);
                diag.help(help);
                diag.emit();
            },
        );
    }
}

#[derive(Eq, PartialEq, Clone, Debug)]
crate struct LangString {
    original: String,
    crate should_panic: bool,
    crate no_run: bool,
    crate ignore: Ignore,
    crate rust: bool,
    crate test_harness: bool,
    crate compile_fail: bool,
    crate error_codes: Vec<String>,
    crate allow_fail: bool,
    crate edition: Option<Edition>,
}

#[derive(Eq, PartialEq, Clone, Debug)]
crate enum Ignore {
    All,
    None,
    Some(Vec<String>),
}

impl Default for LangString {
    fn default() -> Self {
        Self {
            original: String::new(),
            should_panic: false,
            no_run: false,
            ignore: Ignore::None,
            rust: true,
            test_harness: false,
            compile_fail: false,
            error_codes: Vec::new(),
            allow_fail: false,
            edition: None,
        }
    }
}

impl LangString {
    fn parse_without_check(
        string: &str,
        allow_error_code_check: ErrorCodes,
        enable_per_target_ignores: bool,
    ) -> LangString {
        Self::parse(string, allow_error_code_check, enable_per_target_ignores, None)
    }

    fn parse(
        string: &str,
        allow_error_code_check: ErrorCodes,
        enable_per_target_ignores: bool,
        extra: Option<&ExtraInfo<'_, '_>>,
    ) -> LangString {
        let allow_error_code_check = allow_error_code_check.as_bool();
        let mut seen_rust_tags = false;
        let mut seen_other_tags = false;
        let mut data = LangString::default();
        let mut ignores = vec![];

        data.original = string.to_owned();
        let tokens = string.split(|c: char| !(c == '_' || c == '-' || c.is_alphanumeric()));

        for token in tokens {
            match token.trim() {
                "" => {}
                "should_panic" => {
                    data.should_panic = true;
                    seen_rust_tags = !seen_other_tags;
                }
                "no_run" => {
                    data.no_run = true;
                    seen_rust_tags = !seen_other_tags;
                }
                "ignore" => {
                    data.ignore = Ignore::All;
                    seen_rust_tags = !seen_other_tags;
                }
                x if x.starts_with("ignore-") => {
                    if enable_per_target_ignores {
                        ignores.push(x.trim_start_matches("ignore-").to_owned());
                        seen_rust_tags = !seen_other_tags;
                    }
                }
                "allow_fail" => {
                    data.allow_fail = true;
                    seen_rust_tags = !seen_other_tags;
                }
                "rust" => {
                    data.rust = true;
                    seen_rust_tags = true;
                }
                "test_harness" => {
                    data.test_harness = true;
                    seen_rust_tags = !seen_other_tags || seen_rust_tags;
                }
                "compile_fail" => {
                    data.compile_fail = true;
                    seen_rust_tags = !seen_other_tags || seen_rust_tags;
                    data.no_run = true;
                }
                x if x.starts_with("edition") => {
                    data.edition = x[7..].parse::<Edition>().ok();
                }
                x if allow_error_code_check && x.starts_with('E') && x.len() == 5 => {
                    if x[1..].parse::<u32>().is_ok() {
                        data.error_codes.push(x.to_owned());
                        seen_rust_tags = !seen_other_tags || seen_rust_tags;
                    } else {
                        seen_other_tags = true;
                    }
                }
                x if extra.is_some() => {
                    let s = x.to_lowercase();
                    match if s == "compile-fail" || s == "compile_fail" || s == "compilefail" {
                        Some((
                            "compile_fail",
                            "the code block will either not be tested if not marked as a rust one \
                             or won't fail if it compiles successfully",
                        ))
                    } else if s == "should-panic" || s == "should_panic" || s == "shouldpanic" {
                        Some((
                            "should_panic",
                            "the code block will either not be tested if not marked as a rust one \
                             or won't fail if it doesn't panic when running",
                        ))
                    } else if s == "no-run" || s == "no_run" || s == "norun" {
                        Some((
                            "no_run",
                            "the code block will either not be tested if not marked as a rust one \
                             or will be run (which you might not want)",
                        ))
                    } else if s == "allow-fail" || s == "allow_fail" || s == "allowfail" {
                        Some((
                            "allow_fail",
                            "the code block will either not be tested if not marked as a rust one \
                             or will be run (which you might not want)",
                        ))
                    } else if s == "test-harness" || s == "test_harness" || s == "testharness" {
                        Some((
                            "test_harness",
                            "the code block will either not be tested if not marked as a rust one \
                             or the code will be wrapped inside a main function",
                        ))
                    } else {
                        None
                    } {
                        Some((flag, help)) => {
                            if let Some(ref extra) = extra {
                                extra.error_invalid_codeblock_attr(
                                    &format!("unknown attribute `{}`. Did you mean `{}`?", x, flag),
                                    help,
                                );
                            }
                        }
                        None => {}
                    }
                    seen_other_tags = true;
                }
                _ => seen_other_tags = true,
            }
        }
        // ignore-foo overrides ignore
        if !ignores.is_empty() {
            data.ignore = Ignore::Some(ignores);
        }

        data.rust &= !seen_other_tags || seen_rust_tags;

        data
    }
}

impl Markdown<'_> {
    pub fn into_string(self) -> String {
        let Markdown(md, links, mut ids, codes, edition, playground) = self;

        // This is actually common enough to special-case
        if md.is_empty() {
            return String::new();
        }
        let mut replacer = |broken_link: BrokenLink<'_>| {
            if let Some(link) =
                links.iter().find(|link| &*link.original_text == broken_link.reference)
            {
                Some((link.href.as_str().into(), link.new_text.as_str().into()))
            } else {
                None
            }
        };

        let p = Parser::new_with_broken_link_callback(md, opts(), Some(&mut replacer));
        let p = p.into_offset_iter();

        let mut s = String::with_capacity(md.len() * 3 / 2);

        let p = HeadingLinks::new(p, None, &mut ids);
        let p = Footnotes::new(p);
        let p = LinkReplacer::new(p.map(|(ev, _)| ev), links);
        let p = CodeBlocks::new(p, codes, edition, playground);
        html::push_html(&mut s, p);

        s
    }
}

impl MarkdownWithToc<'_> {
    crate fn into_string(self) -> String {
        let MarkdownWithToc(md, mut ids, codes, edition, playground) = self;

        let p = Parser::new_ext(md, opts()).into_offset_iter();

        let mut s = String::with_capacity(md.len() * 3 / 2);

        let mut toc = TocBuilder::new();

        {
            let p = HeadingLinks::new(p, Some(&mut toc), &mut ids);
            let p = Footnotes::new(p);
            let p = CodeBlocks::new(p.map(|(ev, _)| ev), codes, edition, playground);
            html::push_html(&mut s, p);
        }

        format!("<nav id=\"TOC\">{}</nav>{}", toc.into_toc().print(), s)
    }
}

impl MarkdownHtml<'_> {
    crate fn into_string(self) -> String {
        let MarkdownHtml(md, mut ids, codes, edition, playground) = self;

        // This is actually common enough to special-case
        if md.is_empty() {
            return String::new();
        }
        let p = Parser::new_ext(md, opts()).into_offset_iter();

        // Treat inline HTML as plain text.
        let p = p.map(|event| match event.0 {
            Event::Html(text) => (Event::Text(text), event.1),
            _ => event,
        });

        let mut s = String::with_capacity(md.len() * 3 / 2);

        let p = HeadingLinks::new(p, None, &mut ids);
        let p = Footnotes::new(p);
        let p = CodeBlocks::new(p.map(|(ev, _)| ev), codes, edition, playground);
        html::push_html(&mut s, p);

        s
    }
}

impl MarkdownSummaryLine<'_> {
    crate fn into_string(self) -> String {
        let MarkdownSummaryLine(md, links) = self;
        // This is actually common enough to special-case
        if md.is_empty() {
            return String::new();
        }

        let mut replacer = |broken_link: BrokenLink<'_>| {
            if let Some(link) =
                links.iter().find(|link| &*link.original_text == broken_link.reference)
            {
                Some((link.href.as_str().into(), link.new_text.as_str().into()))
            } else {
                None
            }
        };

        let p = Parser::new_with_broken_link_callback(md, summary_opts(), Some(&mut replacer));

        let mut s = String::new();

        html::push_html(&mut s, LinkReplacer::new(SummaryLine::new(p), links));

        s
    }
}

/// Renders a subset of Markdown in the first paragraph of the provided Markdown.
///
/// - *Italics*, **bold**, and `inline code` styles **are** rendered.
/// - Headings and links are stripped (though the text *is* rendered).
/// - HTML, code blocks, and everything else are ignored.
///
/// Returns a tuple of the rendered HTML string and whether the output was shortened
/// due to the provided `length_limit`.
fn markdown_summary_with_limit(md: &str, length_limit: usize) -> (String, bool) {
    if md.is_empty() {
        return (String::new(), false);
    }

    let mut s = String::with_capacity(md.len() * 3 / 2);
    let mut text_length = 0;
    let mut stopped_early = false;

    fn push(s: &mut String, text_length: &mut usize, text: &str) {
        s.push_str(text);
        *text_length += text.len();
    };

    'outer: for event in Parser::new_ext(md, summary_opts()) {
        match &event {
            Event::Text(text) => {
                for word in text.split_inclusive(char::is_whitespace) {
                    if text_length + word.len() >= length_limit {
                        stopped_early = true;
                        break 'outer;
                    }

                    push(&mut s, &mut text_length, word);
                }
            }
            Event::Code(code) => {
                if text_length + code.len() >= length_limit {
                    stopped_early = true;
                    break;
                }

                s.push_str("<code>");
                push(&mut s, &mut text_length, code);
                s.push_str("</code>");
            }
            Event::Start(tag) => match tag {
                Tag::Emphasis => s.push_str("<em>"),
                Tag::Strong => s.push_str("<strong>"),
                Tag::CodeBlock(..) => break,
                _ => {}
            },
            Event::End(tag) => match tag {
                Tag::Emphasis => s.push_str("</em>"),
                Tag::Strong => s.push_str("</strong>"),
                Tag::Paragraph => break,
                _ => {}
            },
            Event::HardBreak | Event::SoftBreak => {
                if text_length + 1 >= length_limit {
                    stopped_early = true;
                    break;
                }

                push(&mut s, &mut text_length, " ");
            }
            _ => {}
        }
    }

    (s, stopped_early)
}

/// Renders a shortened first paragraph of the given Markdown as a subset of Markdown,
/// making it suitable for contexts like the search index.
///
/// Will shorten to 59 or 60 characters, including an ellipsis (…) if it was shortened.
///
/// See [`markdown_summary_with_limit`] for details about what is rendered and what is not.
crate fn short_markdown_summary(markdown: &str) -> String {
    let (mut s, was_shortened) = markdown_summary_with_limit(markdown, 59);

    if was_shortened {
        s.push('…');
    }

    s
}

/// Renders the first paragraph of the provided markdown as plain text.
/// Useful for alt-text.
///
/// - Headings, links, and formatting are stripped.
/// - Inline code is rendered as-is, surrounded by backticks.
/// - HTML and code blocks are ignored.
crate fn plain_text_summary(md: &str) -> String {
    if md.is_empty() {
        return String::new();
    }

    let mut s = String::with_capacity(md.len() * 3 / 2);

    for event in Parser::new_ext(md, summary_opts()) {
        match &event {
            Event::Text(text) => s.push_str(text),
            Event::Code(code) => {
                s.push('`');
                s.push_str(code);
                s.push('`');
            }
            Event::HardBreak | Event::SoftBreak => s.push(' '),
            Event::Start(Tag::CodeBlock(..)) => break,
            Event::End(Tag::Paragraph) => break,
            _ => (),
        }
    }

    s
}

crate fn markdown_links(md: &str) -> Vec<(String, Range<usize>)> {
    if md.is_empty() {
        return vec![];
    }

    let mut links = vec![];
    // Used to avoid mutable borrow issues in the `push` closure
    // Probably it would be more efficient to use a `RefCell` but it doesn't seem worth the churn.
    let mut shortcut_links = vec![];

    let span_for_link = |link: &str, span: Range<usize>| {
        // Pulldown includes the `[]` as well as the URL. Only highlight the relevant span.
        // NOTE: uses `rfind` in case the title and url are the same: `[Ok][Ok]`
        match md[span.clone()].rfind(link) {
            Some(start) => {
                let start = span.start + start;
                start..start + link.len()
            }
            // This can happen for things other than intra-doc links, like `#1` expanded to `https://github.com/rust-lang/rust/issues/1`.
            None => span,
        }
    };
    let mut push = |link: BrokenLink<'_>| {
        let span = span_for_link(link.reference, link.span);
        shortcut_links.push((link.reference.to_owned(), span));
        None
    };
    let p = Parser::new_with_broken_link_callback(md, opts(), Some(&mut push));

    // There's no need to thread an IdMap through to here because
    // the IDs generated aren't going to be emitted anywhere.
    let mut ids = IdMap::new();
    let iter = Footnotes::new(HeadingLinks::new(p.into_offset_iter(), None, &mut ids));

    for ev in iter {
        if let Event::Start(Tag::Link(_, dest, _)) = ev.0 {
            debug!("found link: {}", dest);
            let span = span_for_link(&dest, ev.1);
            links.push((dest.into_string(), span));
        }
    }

    links.append(&mut shortcut_links);

    links
}

#[derive(Debug)]
crate struct RustCodeBlock {
    /// The range in the markdown that the code block occupies. Note that this includes the fences
    /// for fenced code blocks.
    crate range: Range<usize>,
    /// The range in the markdown that the code within the code block occupies.
    crate code: Range<usize>,
    crate is_fenced: bool,
    crate syntax: Option<String>,
}

/// Returns a range of bytes for each code block in the markdown that is tagged as `rust` or
/// untagged (and assumed to be rust).
crate fn rust_code_blocks(md: &str, extra_info: &ExtraInfo<'_, '_>) -> Vec<RustCodeBlock> {
    let mut code_blocks = vec![];

    if md.is_empty() {
        return code_blocks;
    }

    let mut p = Parser::new_ext(md, opts()).into_offset_iter();

    while let Some((event, offset)) = p.next() {
        if let Event::Start(Tag::CodeBlock(syntax)) = event {
            let (syntax, code_start, code_end, range, is_fenced) = match syntax {
                CodeBlockKind::Fenced(syntax) => {
                    let syntax = syntax.as_ref();
                    let lang_string = if syntax.is_empty() {
                        Default::default()
                    } else {
                        LangString::parse(&*syntax, ErrorCodes::Yes, false, Some(extra_info))
                    };
                    if !lang_string.rust {
                        continue;
                    }
                    let syntax = if syntax.is_empty() { None } else { Some(syntax.to_owned()) };
                    let (code_start, mut code_end) = match p.next() {
                        Some((Event::Text(_), offset)) => (offset.start, offset.end),
                        Some((_, sub_offset)) => {
                            let code = Range { start: sub_offset.start, end: sub_offset.start };
                            code_blocks.push(RustCodeBlock {
                                is_fenced: true,
                                range: offset,
                                code,
                                syntax,
                            });
                            continue;
                        }
                        None => {
                            let code = Range { start: offset.end, end: offset.end };
                            code_blocks.push(RustCodeBlock {
                                is_fenced: true,
                                range: offset,
                                code,
                                syntax,
                            });
                            continue;
                        }
                    };
                    while let Some((Event::Text(_), offset)) = p.next() {
                        code_end = offset.end;
                    }
                    (syntax, code_start, code_end, offset, true)
                }
                CodeBlockKind::Indented => {
                    // The ending of the offset goes too far sometime so we reduce it by one in
                    // these cases.
                    if offset.end > offset.start && md.get(offset.end..=offset.end) == Some(&"\n") {
                        (
                            None,
                            offset.start,
                            offset.end,
                            Range { start: offset.start, end: offset.end - 1 },
                            false,
                        )
                    } else {
                        (None, offset.start, offset.end, offset, false)
                    }
                }
            };

            code_blocks.push(RustCodeBlock {
                is_fenced,
                range,
                code: Range { start: code_start, end: code_end },
                syntax,
            });
        }
    }

    code_blocks
}

#[derive(Clone, Default, Debug)]
pub struct IdMap {
    map: FxHashMap<String, usize>,
}

fn init_id_map() -> FxHashMap<String, usize> {
    let mut map = FxHashMap::default();
    // This is the list of IDs used by rustdoc templates.
    map.insert("mainThemeStyle".to_owned(), 1);
    map.insert("themeStyle".to_owned(), 1);
    map.insert("theme-picker".to_owned(), 1);
    map.insert("theme-choices".to_owned(), 1);
    map.insert("settings-menu".to_owned(), 1);
    map.insert("main".to_owned(), 1);
    map.insert("search".to_owned(), 1);
    map.insert("crate-search".to_owned(), 1);
    map.insert("render-detail".to_owned(), 1);
    map.insert("toggle-all-docs".to_owned(), 1);
    map.insert("all-types".to_owned(), 1);
    map.insert("default-settings".to_owned(), 1);
    // This is the list of IDs used by rustdoc sections.
    map.insert("fields".to_owned(), 1);
    map.insert("variants".to_owned(), 1);
    map.insert("implementors-list".to_owned(), 1);
    map.insert("synthetic-implementors-list".to_owned(), 1);
    map.insert("implementations".to_owned(), 1);
    map.insert("trait-implementations".to_owned(), 1);
    map.insert("synthetic-implementations".to_owned(), 1);
    map.insert("blanket-implementations".to_owned(), 1);
    map.insert("deref-methods".to_owned(), 1);
    map
}

impl IdMap {
    pub fn new() -> Self {
        IdMap { map: init_id_map() }
    }

    crate fn populate<I: IntoIterator<Item = String>>(&mut self, ids: I) {
        for id in ids {
            let _ = self.derive(id);
        }
    }

    crate fn reset(&mut self) {
        self.map = init_id_map();
    }

    crate fn derive(&mut self, candidate: String) -> String {
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
