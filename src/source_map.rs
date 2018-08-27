// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! This module contains utilities that work with the `SourceMap` from `libsyntax`/`syntex_syntax`.
//! This includes extension traits and methods for looking up spans and line ranges for AST nodes.

use config::file_lines::LineRange;
use syntax::source_map::{BytePos, SourceMap, Span};
use visitor::SnippetProvider;

use comment::FindUncommented;

pub trait SpanUtils {
    fn span_after(&self, original: Span, needle: &str) -> BytePos;
    fn span_after_last(&self, original: Span, needle: &str) -> BytePos;
    fn span_before(&self, original: Span, needle: &str) -> BytePos;
    fn opt_span_after(&self, original: Span, needle: &str) -> Option<BytePos>;
    fn opt_span_before(&self, original: Span, needle: &str) -> Option<BytePos>;
}

pub trait LineRangeUtils {
    /// Returns the `LineRange` that corresponds to `span` in `self`.
    ///
    /// # Panics
    ///
    /// Panics if `span` crosses a file boundary, which shouldn't happen.
    fn lookup_line_range(&self, span: Span) -> LineRange;
}

impl<'a> SpanUtils for SnippetProvider<'a> {
    fn span_after(&self, original: Span, needle: &str) -> BytePos {
        self.opt_span_after(original, needle).expect("bad span")
    }

    fn span_after_last(&self, original: Span, needle: &str) -> BytePos {
        let snippet = self.span_to_snippet(original).unwrap();
        let mut offset = 0;

        while let Some(additional_offset) = snippet[offset..].find_uncommented(needle) {
            offset += additional_offset + needle.len();
        }

        original.lo() + BytePos(offset as u32)
    }

    fn span_before(&self, original: Span, needle: &str) -> BytePos {
        self.opt_span_before(original, needle).expect(&format!(
            "bad span: {}: {}",
            needle,
            self.span_to_snippet(original).unwrap()
        ))
    }

    fn opt_span_after(&self, original: Span, needle: &str) -> Option<BytePos> {
        self.opt_span_before(original, needle)
            .map(|bytepos| bytepos + BytePos(needle.len() as u32))
    }

    fn opt_span_before(&self, original: Span, needle: &str) -> Option<BytePos> {
        let snippet = self.span_to_snippet(original)?;
        let offset = snippet.find_uncommented(needle)?;

        Some(original.lo() + BytePos(offset as u32))
    }
}

impl LineRangeUtils for SourceMap {
    fn lookup_line_range(&self, span: Span) -> LineRange {
        let lo = self.lookup_line(span.lo()).unwrap();
        let hi = self.lookup_line(span.hi()).unwrap();

        debug_assert_eq!(
            lo.fm.name, hi.fm.name,
            "span crossed file boundary: lo: {:?}, hi: {:?}",
            lo, hi
        );

        // Line numbers start at 1
        LineRange {
            file: lo.fm.clone(),
            lo: lo.line + 1,
            hi: hi.line + 1,
        }
    }
}
