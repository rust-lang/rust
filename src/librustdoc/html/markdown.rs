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
//! This module implements markdown formatting through the sundown C-library
//! (bundled into the rust runtime). This module self-contains the C bindings
//! and necessary legwork to render markdown, and exposes all of the
//! functionality through a unit-struct, `Markdown`, which has an implementation
//! of `fmt::Show`. Example usage:
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
use std::cast;
use std::fmt;
use std::io;
use std::local_data;
use std::mem;
use std::str;
use std::slice;
use collections::HashMap;

use html::toc::TocBuilder;
use html::highlight;

/// A unit struct which has the `fmt::Show` trait implemented. When
/// formatted, this struct will emit the HTML corresponding to the rendered
/// version of the contained markdown string.
pub struct Markdown<'a>(pub &'a str);
/// A unit struct like `Markdown`, that renders the markdown with a
/// table of contents.
pub struct MarkdownWithToc<'a>(pub &'a str);

static OUTPUT_UNIT: libc::size_t = 64;
static MKDEXT_NO_INTRA_EMPHASIS: libc::c_uint = 1 << 0;
static MKDEXT_TABLES: libc::c_uint = 1 << 1;
static MKDEXT_FENCED_CODE: libc::c_uint = 1 << 2;
static MKDEXT_AUTOLINK: libc::c_uint = 1 << 3;
static MKDEXT_STRIKETHROUGH: libc::c_uint = 1 << 4;

type sd_markdown = libc::c_void;  // this is opaque to us

struct sd_callbacks {
    blockcode: Option<extern "C" fn(*buf, *buf, *buf, *libc::c_void)>,
    blockquote: Option<extern "C" fn(*buf, *buf, *libc::c_void)>,
    blockhtml: Option<extern "C" fn(*buf, *buf, *libc::c_void)>,
    header: Option<extern "C" fn(*buf, *buf, libc::c_int, *libc::c_void)>,
    other: [libc::size_t, ..22],
}

struct html_toc_data {
    header_count: libc::c_int,
    current_level: libc::c_int,
    level_offset: libc::c_int,
}

struct html_renderopt {
    toc_data: html_toc_data,
    flags: libc::c_uint,
    link_attributes: Option<extern "C" fn(*buf, *buf, *libc::c_void)>,
}

struct my_opaque {
    opt: html_renderopt,
    dfltblk: extern "C" fn(*buf, *buf, *buf, *libc::c_void),
    toc_builder: Option<TocBuilder>,
}

struct buf {
    data: *u8,
    size: libc::size_t,
    asize: libc::size_t,
    unit: libc::size_t,
}

// sundown FFI
#[link(name = "sundown", kind = "static")]
extern {
    fn sdhtml_renderer(callbacks: *sd_callbacks,
                       options_ptr: *html_renderopt,
                       render_flags: libc::c_uint);
    fn sd_markdown_new(extensions: libc::c_uint,
                       max_nesting: libc::size_t,
                       callbacks: *sd_callbacks,
                       opaque: *libc::c_void) -> *sd_markdown;
    fn sd_markdown_render(ob: *buf,
                          document: *u8,
                          doc_size: libc::size_t,
                          md: *sd_markdown);
    fn sd_markdown_free(md: *sd_markdown);

    fn bufnew(unit: libc::size_t) -> *buf;
    fn bufputs(b: *buf, c: *libc::c_char);
    fn bufrelease(b: *buf);

}

/// Returns Some(code) if `s` is a line that should be stripped from
/// documentation but used in example code. `code` is the portion of
/// `s` that should be used in tests. (None for lines that should be
/// left as-is.)
fn stripped_filtered_line<'a>(s: &'a str) -> Option<&'a str> {
    let trimmed = s.trim();
    if trimmed.starts_with("# ") {
        Some(trimmed.slice_from(2))
    } else {
        None
    }
}

local_data_key!(used_header_map: HashMap<~str, uint>)

pub fn render(w: &mut io::Writer, s: &str, print_toc: bool) -> fmt::Result {
    extern fn block(ob: *buf, text: *buf, lang: *buf, opaque: *libc::c_void) {
        unsafe {
            let my_opaque: &my_opaque = cast::transmute(opaque);
            slice::raw::buf_as_slice((*text).data, (*text).size as uint, |text| {
                let text = str::from_utf8(text).unwrap();
                let mut lines = text.lines().filter(|l| stripped_filtered_line(*l).is_none());
                let text = lines.collect::<Vec<&str>>().connect("\n");

                let buf = buf {
                    data: text.as_bytes().as_ptr(),
                    size: text.len() as libc::size_t,
                    asize: text.len() as libc::size_t,
                    unit: 0,
                };
                let rendered = if lang.is_null() {
                    false
                } else {
                    slice::raw::buf_as_slice((*lang).data,
                                           (*lang).size as uint, |rlang| {
                        let rlang = str::from_utf8(rlang).unwrap();
                        if rlang.contains("notrust") {
                            (my_opaque.dfltblk)(ob, &buf, lang, opaque);
                            true
                        } else {
                            false
                        }
                    })
                };

                if !rendered {
                    let output = highlight::highlight(text, None).to_c_str();
                    output.with_ref(|r| {
                        bufputs(ob, r)
                    })
                }
            })
        }
    }

    extern fn header(ob: *buf, text: *buf, level: libc::c_int,
                     opaque: *libc::c_void) {
        // sundown does this, we may as well too
        "\n".with_c_str(|p| unsafe { bufputs(ob, p) });

        // Extract the text provided
        let s = if text.is_null() {
            "".to_owned()
        } else {
            unsafe {
                str::raw::from_buf_len((*text).data, (*text).size as uint)
            }
        };

        // Transform the contents of the header into a hyphenated string
        let id = s.words().map(|s| {
            match s.to_ascii_opt() {
                Some(s) => s.to_lower().into_str(),
                None => s.to_owned()
            }
        }).collect::<Vec<~str>>().connect("-");

        let opaque = unsafe {&mut *(opaque as *mut my_opaque)};

        // Make sure our hyphenated ID is unique for this page
        let id = local_data::get_mut(used_header_map, |map| {
            let map = map.unwrap();
            match map.find_mut(&id) {
                None => {}
                Some(a) => { *a += 1; return format!("{}-{}", id, *a - 1) }
            }
            map.insert(id.clone(), 1);
            id.clone()
        });

        let sec = match opaque.toc_builder {
            Some(ref mut builder) => {
                builder.push(level as u32, s.clone(), id.clone())
            }
            None => {""}
        };

        // Render the HTML
        let text = format!(r#"<h{lvl} id="{id}" class='section-header'><a
                           href="\#{id}">{sec_len,plural,=0{}other{{sec} }}{}</a></h{lvl}>"#,
                           s, lvl = level, id = id,
                           sec_len = sec.len(), sec = sec);

        text.with_c_str(|p| unsafe { bufputs(ob, p) });
    }

    // This code is all lifted from examples/sundown.c in the sundown repo
    unsafe {
        let ob = bufnew(OUTPUT_UNIT);
        let extensions = MKDEXT_NO_INTRA_EMPHASIS | MKDEXT_TABLES |
                         MKDEXT_FENCED_CODE | MKDEXT_AUTOLINK |
                         MKDEXT_STRIKETHROUGH;
        let options = html_renderopt {
            toc_data: html_toc_data {
                header_count: 0,
                current_level: 0,
                level_offset: 0,
            },
            flags: 0,
            link_attributes: None,
        };
        let mut callbacks: sd_callbacks = mem::init();

        sdhtml_renderer(&callbacks, &options, 0);
        let mut opaque = my_opaque {
            opt: options,
            dfltblk: callbacks.blockcode.unwrap(),
            toc_builder: if print_toc {Some(TocBuilder::new())} else {None}
        };
        callbacks.blockcode = Some(block);
        callbacks.header = Some(header);
        let markdown = sd_markdown_new(extensions, 16, &callbacks,
                                       &mut opaque as *mut my_opaque as *libc::c_void);


        sd_markdown_render(ob, s.as_ptr(), s.len() as libc::size_t, markdown);
        sd_markdown_free(markdown);

        let mut ret = match opaque.toc_builder {
            Some(b) => write!(w, "<nav id=\"TOC\">{}</nav>", b.into_toc()),
            None => Ok(())
        };

        if ret.is_ok() {
            ret = slice::raw::buf_as_slice((*ob).data, (*ob).size as uint, |buf| {
                w.write(buf)
            });
        }
        bufrelease(ob);
        ret
    }
}

pub fn find_testable_code(doc: &str, tests: &mut ::test::Collector) {
    extern fn block(_ob: *buf, text: *buf, lang: *buf, opaque: *libc::c_void) {
        unsafe {
            if text.is_null() { return }
            let (should_fail, no_run, ignore, notrust) = if lang.is_null() {
                (false, false, false, false)
            } else {
                slice::raw::buf_as_slice((*lang).data,
                                       (*lang).size as uint, |lang| {
                    let s = str::from_utf8(lang).unwrap();
                    (s.contains("should_fail"),
                     s.contains("no_run"),
                     s.contains("ignore"),
                     s.contains("notrust"))
                })
            };
            if notrust { return }
            slice::raw::buf_as_slice((*text).data, (*text).size as uint, |text| {
                let tests = &mut *(opaque as *mut ::test::Collector);
                let text = str::from_utf8(text).unwrap();
                let mut lines = text.lines().map(|l| stripped_filtered_line(l).unwrap_or(l));
                let text = lines.collect::<Vec<&str>>().connect("\n");
                tests.add_test(text, should_fail, no_run, ignore);
            })
        }
    }
    extern fn header(_ob: *buf, text: *buf, level: libc::c_int, opaque: *libc::c_void) {
        unsafe {
            let tests = &mut *(opaque as *mut ::test::Collector);
            if text.is_null() {
                tests.register_header("", level as u32);
            } else {
                slice::raw::buf_as_slice((*text).data, (*text).size as uint, |text| {
                    let text = str::from_utf8(text).unwrap();
                    tests.register_header(text, level as u32);
                })
            }
        }
    }

    unsafe {
        let ob = bufnew(OUTPUT_UNIT);
        let extensions = MKDEXT_NO_INTRA_EMPHASIS | MKDEXT_TABLES |
                         MKDEXT_FENCED_CODE | MKDEXT_AUTOLINK |
                         MKDEXT_STRIKETHROUGH;
        let callbacks = sd_callbacks {
            blockcode: Some(block),
            blockquote: None,
            blockhtml: None,
            header: Some(header),
            other: mem::init()
        };

        let tests = tests as *mut ::test::Collector as *libc::c_void;
        let markdown = sd_markdown_new(extensions, 16, &callbacks, tests);

        sd_markdown_render(ob, doc.as_ptr(), doc.len() as libc::size_t,
                           markdown);
        sd_markdown_free(markdown);
        bufrelease(ob);
    }
}

/// By default this markdown renderer generates anchors for each header in the
/// rendered document. The anchor name is the contents of the header spearated
/// by hyphens, and a task-local map is used to disambiguate among duplicate
/// headers (numbers are appended).
///
/// This method will reset the local table for these headers. This is typically
/// used at the beginning of rendering an entire HTML page to reset from the
/// previous state (if any).
pub fn reset_headers() {
    local_data::set(used_header_map, HashMap::new())
}

impl<'a> fmt::Show for Markdown<'a> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        let Markdown(md) = *self;
        // This is actually common enough to special-case
        if md.len() == 0 { return Ok(()) }
        render(fmt.buf, md.as_slice(), false)
    }
}

impl<'a> fmt::Show for MarkdownWithToc<'a> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        let MarkdownWithToc(md) = *self;
        render(fmt.buf, md.as_slice(), true)
    }
}
