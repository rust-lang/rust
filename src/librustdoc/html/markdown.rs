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
//! This module implements markdown formatting through the hoedown C-library
//! (bundled into the rust runtime). This module self-contains the C bindings
//! and necessary legwork to render markdown, and exposes all of the
//! functionality through a unit-struct, `Markdown`, which has an implementation
//! of `fmt::Display`. Example usage:
//!
//! ```rust,ignore
//! use rustdoc::html::markdown::Markdown;
//!
//! let s = "My *markdown* _text_";
//! let html = format!("{}", Markdown(s));
//! // ... something using html
//! ```

#![allow(non_camel_case_types)]

use libc;
use std::ascii::AsciiExt;
use std::cell::RefCell;
use std::default::Default;
use std::ffi::CString;
use std::fmt;
use std::slice;
use std::str;

use html::render::derive_id;
use html::toc::TocBuilder;
use html::highlight;
use html::escape::Escape;
use test;

/// A unit struct which has the `fmt::Display` trait implemented. When
/// formatted, this struct will emit the HTML corresponding to the rendered
/// version of the contained markdown string.
pub struct Markdown<'a>(pub &'a str);
/// A unit struct like `Markdown`, that renders the markdown with a
/// table of contents.
pub struct MarkdownWithToc<'a>(pub &'a str);

const DEF_OUNIT: libc::size_t = 64;
const HOEDOWN_EXT_NO_INTRA_EMPHASIS: libc::c_uint = 1 << 11;
const HOEDOWN_EXT_TABLES: libc::c_uint = 1 << 0;
const HOEDOWN_EXT_FENCED_CODE: libc::c_uint = 1 << 1;
const HOEDOWN_EXT_AUTOLINK: libc::c_uint = 1 << 3;
const HOEDOWN_EXT_STRIKETHROUGH: libc::c_uint = 1 << 4;
const HOEDOWN_EXT_SUPERSCRIPT: libc::c_uint = 1 << 8;
const HOEDOWN_EXT_FOOTNOTES: libc::c_uint = 1 << 2;

const HOEDOWN_EXTENSIONS: libc::c_uint =
    HOEDOWN_EXT_NO_INTRA_EMPHASIS | HOEDOWN_EXT_TABLES |
    HOEDOWN_EXT_FENCED_CODE | HOEDOWN_EXT_AUTOLINK |
    HOEDOWN_EXT_STRIKETHROUGH | HOEDOWN_EXT_SUPERSCRIPT |
    HOEDOWN_EXT_FOOTNOTES;

enum hoedown_document {}

type blockcodefn = extern "C" fn(*mut hoedown_buffer, *const hoedown_buffer,
                                 *const hoedown_buffer, *const hoedown_renderer_data);

type blockquotefn = extern "C" fn(*mut hoedown_buffer, *const hoedown_buffer,
                                  *const hoedown_renderer_data);

type headerfn = extern "C" fn(*mut hoedown_buffer, *const hoedown_buffer,
                              libc::c_int, *const hoedown_renderer_data);

type blockhtmlfn = extern "C" fn(*mut hoedown_buffer, *const hoedown_buffer,
                                 *const hoedown_renderer_data);

type codespanfn = extern "C" fn(*mut hoedown_buffer, *const hoedown_buffer,
                                *const hoedown_renderer_data) -> libc::c_int;

type linkfn = extern "C" fn (*mut hoedown_buffer, *const hoedown_buffer,
                             *const hoedown_buffer, *const hoedown_buffer,
                             *const hoedown_renderer_data) -> libc::c_int;

type entityfn = extern "C" fn (*mut hoedown_buffer, *const hoedown_buffer,
                               *const hoedown_renderer_data);

type normaltextfn = extern "C" fn(*mut hoedown_buffer, *const hoedown_buffer,
                                  *const hoedown_renderer_data);

#[repr(C)]
struct hoedown_renderer_data {
    opaque: *mut libc::c_void,
}

#[repr(C)]
struct hoedown_renderer {
    opaque: *mut libc::c_void,

    blockcode: Option<blockcodefn>,
    blockquote: Option<blockquotefn>,
    header: Option<headerfn>,

    other_block_level_callbacks: [libc::size_t; 11],

    blockhtml: Option<blockhtmlfn>,

    /* span level callbacks - NULL or return 0 prints the span verbatim */
    autolink: libc::size_t, // unused
    codespan: Option<codespanfn>,
    other_span_level_callbacks_1: [libc::size_t; 7],
    link: Option<linkfn>,
    other_span_level_callbacks_2: [libc::size_t; 6],

    /* low level callbacks - NULL copies input directly into the output */
    entity: Option<entityfn>,
    normal_text: Option<normaltextfn>,

    /* header and footer */
    other_callbacks: [libc::size_t; 2],
}

#[repr(C)]
struct hoedown_html_renderer_state {
    opaque: *mut libc::c_void,
    toc_data: html_toc_data,
    flags: libc::c_uint,
    link_attributes: Option<extern "C" fn(*mut hoedown_buffer,
                                          *const hoedown_buffer,
                                          *const hoedown_renderer_data)>,
}

#[repr(C)]
struct html_toc_data {
    header_count: libc::c_int,
    current_level: libc::c_int,
    level_offset: libc::c_int,
    nesting_level: libc::c_int,
}

struct MyOpaque {
    dfltblk: extern "C" fn(*mut hoedown_buffer, *const hoedown_buffer,
                           *const hoedown_buffer, *const hoedown_renderer_data),
    toc_builder: Option<TocBuilder>,
}

#[repr(C)]
struct hoedown_buffer {
    data: *const u8,
    size: libc::size_t,
    asize: libc::size_t,
    unit: libc::size_t,
}

// hoedown FFI
#[link(name = "hoedown", kind = "static")]
extern {
    fn hoedown_html_renderer_new(render_flags: libc::c_uint,
                                 nesting_level: libc::c_int)
        -> *mut hoedown_renderer;
    fn hoedown_html_renderer_free(renderer: *mut hoedown_renderer);

    fn hoedown_document_new(rndr: *const hoedown_renderer,
                            extensions: libc::c_uint,
                            max_nesting: libc::size_t) -> *mut hoedown_document;
    fn hoedown_document_render(doc: *mut hoedown_document,
                               ob: *mut hoedown_buffer,
                               document: *const u8,
                               doc_size: libc::size_t);
    fn hoedown_document_free(md: *mut hoedown_document);

    fn hoedown_buffer_new(unit: libc::size_t) -> *mut hoedown_buffer;
    fn hoedown_buffer_put(b: *mut hoedown_buffer, c: *const libc::c_char,
                          n: libc::size_t);
    fn hoedown_buffer_puts(b: *mut hoedown_buffer, c: *const libc::c_char);
    fn hoedown_buffer_free(b: *mut hoedown_buffer);

}

// hoedown_buffer helpers
impl hoedown_buffer {
    fn as_bytes(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(self.data, self.size as usize) }
    }
}

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

thread_local!(pub static PLAYGROUND_KRATE: RefCell<Option<Option<String>>> = {
    RefCell::new(None)
});

pub fn render(w: &mut fmt::Formatter, s: &str, print_toc: bool) -> fmt::Result {
    extern fn block(ob: *mut hoedown_buffer, orig_text: *const hoedown_buffer,
                    lang: *const hoedown_buffer, data: *const hoedown_renderer_data) {
        unsafe {
            if orig_text.is_null() { return }

            let opaque = (*data).opaque as *mut hoedown_html_renderer_state;
            let my_opaque: &MyOpaque = &*((*opaque).opaque as *const MyOpaque);
            let text = (*orig_text).as_bytes();
            let origtext = str::from_utf8(text).unwrap();
            debug!("docblock: ==============\n{:?}\n=======", text);
            let rendered = if lang.is_null() {
                false
            } else {
                let rlang = (*lang).as_bytes();
                let rlang = str::from_utf8(rlang).unwrap();
                if !LangString::parse(rlang).rust {
                    (my_opaque.dfltblk)(ob, orig_text, lang,
                                        opaque as *const hoedown_renderer_data);
                    true
                } else {
                    false
                }
            };

            let lines = origtext.lines().filter(|l| {
                stripped_filtered_line(*l).is_none()
            });
            let text = lines.collect::<Vec<&str>>().join("\n");
            if rendered { return }
            PLAYGROUND_KRATE.with(|krate| {
                let mut s = String::new();
                krate.borrow().as_ref().map(|krate| {
                    let test = origtext.lines().map(|l| {
                        stripped_filtered_line(l).unwrap_or(l)
                    }).collect::<Vec<&str>>().join("\n");
                    let krate = krate.as_ref().map(|s| &**s);
                    let test = test::maketest(&test, krate, false,
                                              &Default::default());
                    s.push_str(&format!("<span class='rusttest'>{}</span>", Escape(&test)));
                });
                s.push_str(&highlight::highlight(&text,
                                                 Some("rust-example-rendered"),
                                                 None));
                let output = CString::new(s).unwrap();
                hoedown_buffer_puts(ob, output.as_ptr());
            })
        }
    }

    extern fn header(ob: *mut hoedown_buffer, text: *const hoedown_buffer,
                     level: libc::c_int, data: *const hoedown_renderer_data) {
        // hoedown does this, we may as well too
        unsafe { hoedown_buffer_puts(ob, "\n\0".as_ptr() as *const _); }

        // Extract the text provided
        let s = if text.is_null() {
            "".to_owned()
        } else {
            let s = unsafe { (*text).as_bytes() };
            str::from_utf8(&s).unwrap().to_owned()
        };

        // Discard '<em>', '<code>' tags and some escaped characters,
        // transform the contents of the header into a hyphenated string
        // without non-alphanumeric characters other than '-' and '_'.
        //
        // This is a terrible hack working around how hoedown gives us rendered
        // html for text rather than the raw text.
        let mut id = s.clone();
        let repl_sub = vec!["<em>", "</em>", "<code>", "</code>",
                            "<strong>", "</strong>",
                            "&lt;", "&gt;", "&amp;", "&#39;", "&quot;"];
        for sub in repl_sub {
            id = id.replace(sub, "");
        }
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

        let opaque = unsafe { (*data).opaque as *mut hoedown_html_renderer_state };
        let opaque = unsafe { &mut *((*opaque).opaque as *mut MyOpaque) };

        let id = derive_id(id);

        let sec = opaque.toc_builder.as_mut().map_or("".to_owned(), |builder| {
            format!("{} ", builder.push(level as u32, s.clone(), id.clone()))
        });

        // Render the HTML
        let text = format!("<h{lvl} id='{id}' class='section-header'>\
                           <a href='#{id}'>{sec}{}</a></h{lvl}>",
                           s, lvl = level, id = id, sec = sec);

        let text = CString::new(text).unwrap();
        unsafe { hoedown_buffer_puts(ob, text.as_ptr()) }
    }

    extern fn codespan(
        ob: *mut hoedown_buffer,
        text: *const hoedown_buffer,
        _: *const hoedown_renderer_data,
    ) -> libc::c_int {
        let content = if text.is_null() {
            "".to_owned()
        } else {
            let bytes = unsafe { (*text).as_bytes() };
            let s = str::from_utf8(bytes).unwrap();
            collapse_whitespace(s)
        };

        let content = format!("<code>{}</code>", Escape(&content));
        let element = CString::new(content).unwrap();
        unsafe { hoedown_buffer_puts(ob, element.as_ptr()); }
        // Return anything except 0, which would mean "also print the code span verbatim".
        1
    }

    unsafe {
        let ob = hoedown_buffer_new(DEF_OUNIT);
        let renderer = hoedown_html_renderer_new(0, 0);
        let mut opaque = MyOpaque {
            dfltblk: (*renderer).blockcode.unwrap(),
            toc_builder: if print_toc {Some(TocBuilder::new())} else {None}
        };
        (*((*renderer).opaque as *mut hoedown_html_renderer_state)).opaque
                = &mut opaque as *mut _ as *mut libc::c_void;
        (*renderer).blockcode = Some(block);
        (*renderer).header = Some(header);
        (*renderer).codespan = Some(codespan);

        let document = hoedown_document_new(renderer, HOEDOWN_EXTENSIONS, 16);
        hoedown_document_render(document, ob, s.as_ptr(),
                                s.len() as libc::size_t);
        hoedown_document_free(document);

        hoedown_html_renderer_free(renderer);

        let mut ret = opaque.toc_builder.map_or(Ok(()), |builder| {
            write!(w, "<nav id=\"TOC\">{}</nav>", builder.into_toc())
        });

        if ret.is_ok() {
            let buf = (*ob).as_bytes();
            ret = w.write_str(str::from_utf8(buf).unwrap());
        }
        hoedown_buffer_free(ob);
        ret
    }
}

pub fn find_testable_code(doc: &str, tests: &mut ::test::Collector) {
    extern fn block(_ob: *mut hoedown_buffer,
                    text: *const hoedown_buffer,
                    lang: *const hoedown_buffer,
                    data: *const hoedown_renderer_data) {
        unsafe {
            if text.is_null() { return }
            let block_info = if lang.is_null() {
                LangString::all_false()
            } else {
                let lang = (*lang).as_bytes();
                let s = str::from_utf8(lang).unwrap();
                LangString::parse(s)
            };
            if !block_info.rust { return }
            let text = (*text).as_bytes();
            let opaque = (*data).opaque as *mut hoedown_html_renderer_state;
            let tests = &mut *((*opaque).opaque as *mut ::test::Collector);
            let text = str::from_utf8(text).unwrap();
            let lines = text.lines().map(|l| {
                stripped_filtered_line(l).unwrap_or(l)
            });
            let text = lines.collect::<Vec<&str>>().join("\n");
            tests.add_test(text.to_owned(),
                           block_info.should_panic, block_info.no_run,
                           block_info.ignore, block_info.test_harness);
        }
    }

    extern fn header(_ob: *mut hoedown_buffer,
                     text: *const hoedown_buffer,
                     level: libc::c_int, data: *const hoedown_renderer_data) {
        unsafe {
            let opaque = (*data).opaque as *mut hoedown_html_renderer_state;
            let tests = &mut *((*opaque).opaque as *mut ::test::Collector);
            if text.is_null() {
                tests.register_header("", level as u32);
            } else {
                let text = (*text).as_bytes();
                let text = str::from_utf8(text).unwrap();
                tests.register_header(text, level as u32);
            }
        }
    }

    unsafe {
        let ob = hoedown_buffer_new(DEF_OUNIT);
        let renderer = hoedown_html_renderer_new(0, 0);
        (*renderer).blockcode = Some(block);
        (*renderer).header = Some(header);
        (*((*renderer).opaque as *mut hoedown_html_renderer_state)).opaque
                = tests as *mut _ as *mut libc::c_void;

        let document = hoedown_document_new(renderer, HOEDOWN_EXTENSIONS, 16);
        hoedown_document_render(document, ob, doc.as_ptr(),
                                doc.len() as libc::size_t);
        hoedown_document_free(document);

        hoedown_html_renderer_free(renderer);
        hoedown_buffer_free(ob);
    }
}

#[derive(Eq, PartialEq, Clone, Debug)]
struct LangString {
    should_panic: bool,
    no_run: bool,
    ignore: bool,
    rust: bool,
    test_harness: bool,
}

impl LangString {
    fn all_false() -> LangString {
        LangString {
            should_panic: false,
            no_run: false,
            ignore: false,
            rust: true,  // NB This used to be `notrust = false`
            test_harness: false,
        }
    }

    fn parse(string: &str) -> LangString {
        let mut seen_rust_tags = false;
        let mut seen_other_tags = false;
        let mut data = LangString::all_false();

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
                "test_harness" => { data.test_harness = true; seen_rust_tags = true; }
                _ => { seen_other_tags = true }
            }
        }

        data.rust &= !seen_other_tags || seen_rust_tags;

        data
    }
}

impl<'a> fmt::Display for Markdown<'a> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        let Markdown(md) = *self;
        // This is actually common enough to special-case
        if md.is_empty() { return Ok(()) }
        render(fmt, md, false)
    }
}

impl<'a> fmt::Display for MarkdownWithToc<'a> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        let MarkdownWithToc(md) = *self;
        render(fmt, md, true)
    }
}

pub fn plain_summary_line(md: &str) -> String {
    extern fn link(_ob: *mut hoedown_buffer,
                       _link: *const hoedown_buffer,
                       _title: *const hoedown_buffer,
                       content: *const hoedown_buffer,
                       data: *const hoedown_renderer_data) -> libc::c_int
    {
        unsafe {
            if !content.is_null() && (*content).size > 0 {
                let ob = (*data).opaque as *mut hoedown_buffer;
                hoedown_buffer_put(ob, (*content).data as *const libc::c_char,
                                   (*content).size);
            }
        }
        1
    }

    extern fn normal_text(_ob: *mut hoedown_buffer,
                              text: *const hoedown_buffer,
                              data: *const hoedown_renderer_data)
    {
        unsafe {
            let ob = (*data).opaque as *mut hoedown_buffer;
            hoedown_buffer_put(ob, (*text).data as *const libc::c_char,
                               (*text).size);
        }
    }

    unsafe {
        let ob = hoedown_buffer_new(DEF_OUNIT);
        let mut plain_renderer: hoedown_renderer = ::std::mem::zeroed();
        let renderer: *mut hoedown_renderer = &mut plain_renderer;
        (*renderer).opaque = ob as *mut libc::c_void;
        (*renderer).link = Some(link);
        (*renderer).normal_text = Some(normal_text);

        let document = hoedown_document_new(renderer, HOEDOWN_EXTENSIONS, 16);
        hoedown_document_render(document, ob, md.as_ptr(),
                                md.len() as libc::size_t);
        hoedown_document_free(document);
        let plain_slice = (*ob).as_bytes();
        let plain = str::from_utf8(plain_slice).unwrap_or("").to_owned();
        hoedown_buffer_free(ob);
        plain
    }
}

#[cfg(test)]
mod tests {
    use super::{LangString, Markdown};
    use super::plain_summary_line;
    use html::render::reset_ids;

    #[test]
    fn test_lang_string_parse() {
        fn t(s: &str,
            should_panic: bool, no_run: bool, ignore: bool, rust: bool, test_harness: bool) {
            assert_eq!(LangString::parse(s), LangString {
                should_panic: should_panic,
                no_run: no_run,
                ignore: ignore,
                rust: rust,
                test_harness: test_harness,
            })
        }

        // marker                | should_panic| no_run | ignore | rust | test_harness
        t("",                      false,        false,   false,   true,  false);
        t("rust",                  false,        false,   false,   true,  false);
        t("sh",                    false,        false,   false,   false, false);
        t("ignore",                false,        false,   true,    true,  false);
        t("should_panic",          true,         false,   false,   true,  false);
        t("no_run",                false,        true,    false,   true,  false);
        t("test_harness",          false,        false,   false,   true,  true);
        t("{.no_run .example}",    false,        true,    false,   true,  false);
        t("{.sh .should_panic}",   true,         false,   false,   true,  false);
        t("{.example .rust}",      false,        false,   false,   true,  false);
        t("{.test_harness .rust}", false,        false,   false,   true,  true);
    }

    #[test]
    fn issue_17736() {
        let markdown = "# title";
        format!("{}", Markdown(markdown));
        reset_ids();
    }

    #[test]
    fn test_header() {
        fn t(input: &str, expect: &str) {
            let output = format!("{}", Markdown(input));
            assert_eq!(output, expect);
            reset_ids();
        }

        t("# Foo bar", "\n<h1 id='foo-bar' class='section-header'>\
          <a href='#foo-bar'>Foo bar</a></h1>");
        t("## Foo-bar_baz qux", "\n<h2 id='foo-bar_baz-qux' class=\'section-\
          header'><a href='#foo-bar_baz-qux'>Foo-bar_baz qux</a></h2>");
        t("### **Foo** *bar* baz!?!& -_qux_-%",
          "\n<h3 id='foo-bar-baz--_qux_-' class='section-header'>\
          <a href='#foo-bar-baz--_qux_-'><strong>Foo</strong> \
          <em>bar</em> baz!?!&amp; -_qux_-%</a></h3>");
        t("####**Foo?** & \\*bar?!*  _`baz`_ ❤ #qux",
          "\n<h4 id='foo--bar--baz--qux' class='section-header'>\
          <a href='#foo--bar--baz--qux'><strong>Foo?</strong> &amp; *bar?!*  \
          <em><code>baz</code></em> ❤ #qux</a></h4>");
    }

    #[test]
    fn test_header_ids_multiple_blocks() {
        fn t(input: &str, expect: &str) {
            let output = format!("{}", Markdown(input));
            assert_eq!(output, expect);
        }

        let test = || {
            t("# Example", "\n<h1 id='example' class='section-header'>\
              <a href='#example'>Example</a></h1>");
            t("# Panics", "\n<h1 id='panics' class='section-header'>\
              <a href='#panics'>Panics</a></h1>");
            t("# Example", "\n<h1 id='example-1' class='section-header'>\
              <a href='#example-1'>Example</a></h1>");
            t("# Main", "\n<h1 id='main-1' class='section-header'>\
              <a href='#main-1'>Main</a></h1>");
            t("# Example", "\n<h1 id='example-2' class='section-header'>\
              <a href='#example-2'>Example</a></h1>");
            t("# Panics", "\n<h1 id='panics-1' class='section-header'>\
              <a href='#panics-1'>Panics</a></h1>");
        };
        test();
        reset_ids();
        test();
    }

    #[test]
    fn test_plain_summary_line() {
        fn t(input: &str, expect: &str) {
            let output = plain_summary_line(input);
            assert_eq!(output, expect);
        }

        t("hello [Rust](https://www.rust-lang.org) :)", "hello Rust :)");
        t("code `let x = i32;` ...", "code `let x = i32;` ...");
        t("type `Type<'static>` ...", "type `Type<'static>` ...");
        t("# top header", "top header");
        t("## header", "header");
    }
}
