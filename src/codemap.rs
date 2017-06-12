// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! This module contains utilities that work with the `CodeMap` from libsyntax / syntex_syntax.
//! This includes extension traits and methods for looking up spans and line ranges for AST nodes.

use std::rc::Rc;

use syntax::codemap::{BytePos, CodeMap, FileMap, Span};

use comment::FindUncommented;

/// A range of lines in a file, inclusive of both ends.
pub struct LineRange {
    pub file: Rc<FileMap>,
    pub lo: usize,
    pub hi: usize,
}

impl LineRange {
    pub fn file_name(&self) -> &str {
        self.file.as_ref().name.as_str()
    }
}

pub trait SpanUtils {
    fn span_after(&self, original: Span, needle: &str) -> BytePos;
    fn span_after_last(&self, original: Span, needle: &str) -> BytePos;
    fn span_before(&self, original: Span, needle: &str) -> BytePos;
}

pub trait LineRangeUtils {
    /// Returns the `LineRange` that corresponds to `span` in `self`.
    ///
    /// # Panics
    ///
    /// Panics if `span` crosses a file boundary, which shouldn't happen.
    fn lookup_line_range(&self, span: Span) -> LineRange;
}

impl SpanUtils for CodeMap {
    fn span_after(&self, original: Span, needle: &str) -> BytePos {
        let snippet = self.span_to_snippet(original).unwrap();
        let offset = snippet.find_uncommented(needle).unwrap() + needle.len();

        original.lo + BytePos(offset as u32)
    }

    fn span_after_last(&self, original: Span, needle: &str) -> BytePos {
        let snippet = self.span_to_snippet(original).unwrap();
        let mut offset = 0;

        while let Some(additional_offset) = snippet[offset..].find_uncommented(needle) {
            offset += additional_offset + needle.len();
        }

        original.lo + BytePos(offset as u32)
    }

    fn span_before(&self, original: Span, needle: &str) -> BytePos {
        let snippet = self.span_to_snippet(original).unwrap();
        let offset = snippet.find_uncommented(needle).unwrap();

        original.lo + BytePos(offset as u32)
    }
}

impl LineRangeUtils for CodeMap {
    fn lookup_line_range(&self, span: Span) -> LineRange {
        let lo = self.lookup_char_pos(span.lo);
        let hi = self.lookup_char_pos(span.hi);

        assert!(
            lo.file.name == hi.file.name,
            "span crossed file boundary: lo: {:?}, hi: {:?}",
            lo,
            hi
        );

        LineRange {
            file: lo.file.clone(),
            lo: lo.line,
            hi: hi.line,
        }
    }
}
