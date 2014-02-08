// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
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

use std::cast;
use std::fmt;
use std::io;
use std::libc;
use std::str;
use std::unstable::intrinsics;
use std::vec;

/// A unit struct which has the `fmt::Show` trait implemented. When
/// formatted, this struct will emit the HTML corresponding to the rendered
/// version of the contained markdown string.
pub struct Markdown<'a>(&'a str);

static OUTPUT_UNIT: libc::size_t = 64;
static MKDEXT_NO_INTRA_EMPHASIS: libc::c_uint = 1 << 0;
static MKDEXT_TABLES: libc::c_uint = 1 << 1;
static MKDEXT_FENCED_CODE: libc::c_uint = 1 << 2;
static MKDEXT_AUTOLINK: libc::c_uint = 1 << 3;
static MKDEXT_STRIKETHROUGH: libc::c_uint = 1 << 4;

type sd_markdown = libc::c_void;  // this is opaque to us

struct sd_callbacks {
    blockcode: extern "C" fn(*buf, *buf, *buf, *libc::c_void),
    other: [libc::size_t, ..25],
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

pub fn render(w: &mut io::Writer, s: &str) -> fmt::Result {
    extern fn block(ob: *buf, text: *buf, lang: *buf, opaque: *libc::c_void) {
        unsafe {
            let my_opaque: &my_opaque = cast::transmute(opaque);
            vec::raw::buf_as_slice((*text).data, (*text).size as uint, |text| {
                let text = str::from_utf8(text).unwrap();
                let mut lines = text.lines().filter(|l| stripped_filtered_line(*l).is_none());
                let text = lines.to_owned_vec().connect("\n");

                let buf = buf {
                    data: text.as_bytes().as_ptr(),
                    size: text.len() as libc::size_t,
                    asize: text.len() as libc::size_t,
                    unit: 0,
                };
                (my_opaque.dfltblk)(ob, &buf, lang, opaque);
            })
        }
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
        let mut callbacks: sd_callbacks = intrinsics::init();

        sdhtml_renderer(&callbacks, &options, 0);
        let opaque = my_opaque {
            opt: options,
            dfltblk: callbacks.blockcode,
        };
        callbacks.blockcode = block;
        let markdown = sd_markdown_new(extensions, 16, &callbacks,
                                       &opaque as *my_opaque as *libc::c_void);


        sd_markdown_render(ob, s.as_ptr(), s.len() as libc::size_t, markdown);
        sd_markdown_free(markdown);

        let ret = vec::raw::buf_as_slice((*ob).data, (*ob).size as uint, |buf| {
            w.write(buf)
        });

        bufrelease(ob);
        ret
    }
}

pub fn find_testable_code(doc: &str, tests: &mut ::test::Collector) {
    extern fn block(_ob: *buf, text: *buf, lang: *buf, opaque: *libc::c_void) {
        unsafe {
            if text.is_null() || lang.is_null() { return }
            let (test, shouldfail, ignore) =
                vec::raw::buf_as_slice((*lang).data,
                                       (*lang).size as uint, |lang| {
                    let s = str::from_utf8(lang).unwrap();
                    (s.contains("rust"), s.contains("should_fail"),
                     s.contains("ignore"))
                });
            if !test { return }
            vec::raw::buf_as_slice((*text).data, (*text).size as uint, |text| {
                let tests: &mut ::test::Collector = intrinsics::transmute(opaque);
                let text = str::from_utf8(text).unwrap();
                let mut lines = text.lines().map(|l| stripped_filtered_line(l).unwrap_or(l));
                let text = lines.to_owned_vec().connect("\n");
                tests.add_test(text, ignore, shouldfail);
            })
        }
    }

    unsafe {
        let ob = bufnew(OUTPUT_UNIT);
        let extensions = MKDEXT_NO_INTRA_EMPHASIS | MKDEXT_TABLES |
                         MKDEXT_FENCED_CODE | MKDEXT_AUTOLINK |
                         MKDEXT_STRIKETHROUGH;
        let callbacks = sd_callbacks {
            blockcode: block,
            other: intrinsics::init()
        };

        let tests = tests as *mut ::test::Collector as *libc::c_void;
        let markdown = sd_markdown_new(extensions, 16, &callbacks, tests);

        sd_markdown_render(ob, doc.as_ptr(), doc.len() as libc::size_t,
                           markdown);
        sd_markdown_free(markdown);
        bufrelease(ob);
    }
}

impl<'a> fmt::Show for Markdown<'a> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        let Markdown(md) = *self;
        // This is actually common enough to special-case
        if md.len() == 0 { return Ok(()) }
        render(fmt.buf, md.as_slice())
    }
}
