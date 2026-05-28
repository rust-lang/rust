//! This module contains utilities that work with the `SourceMap` from `libsyntax`/`syntex_syntax`.
//! This includes extension traits and methods for looking up spans and line ranges for AST nodes.

use rustc_span::{BytePos, Span};

use crate::comment::FindUncommented;
use crate::config::file_lines::LineRange;
use crate::visitor::SnippetProvider;

pub(crate) trait SpanUtils {
    fn span_after(&self, original: Span, needle: &str) -> BytePos;
    fn span_after_last(&self, original: Span, needle: &str) -> BytePos;
    fn span_before(&self, original: Span, needle: &str) -> BytePos;
    fn span_before_last(&self, original: Span, needle: &str) -> BytePos;
    fn opt_span_after(&self, original: Span, needle: &str) -> Option<BytePos>;
    fn opt_span_before(&self, original: Span, needle: &str) -> Option<BytePos>;
}

pub(crate) trait LineRangeUtils {
    /// Returns the `LineRange` that corresponds to `span` in `self`.
    ///
    /// # Panics
    ///
    /// Panics if `span` crosses a file boundary, which shouldn't happen.
    fn lookup_line_range(&self, span: Span) -> LineRange;
}

impl SpanUtils for SnippetProvider {
    fn span_after(&self, original: Span, needle: &str) -> BytePos {
        self.opt_span_after(original, needle).unwrap_or_else(|| {
            panic!(
                "bad span: `{}`: `{}`",
                needle,
                self.span_to_snippet(original).unwrap()
            )
        })
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
        self.opt_span_before(original, needle).unwrap_or_else(|| {
            panic!(
                "bad span: `{}`: `{}`",
                needle,
                self.span_to_snippet(original).unwrap()
            )
        })
    }

    fn span_before_last(&self, original: Span, needle: &str) -> BytePos {
        let snippet = self.span_to_snippet(original).unwrap();
        let mut offset = 0;

        while let Some(additional_offset) = snippet[offset..].find_uncommented(needle) {
            offset += additional_offset + needle.len();
        }

        original.lo() + BytePos(offset as u32 - 1)
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
