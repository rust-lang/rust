use syntax::codemap::{BytePos, CodeMap, Span};

use comment::FindUncommented;

pub trait SpanUtils {
    fn span_after(&self, original: Span, needle: &str) -> BytePos;
    fn span_after_last(&self, original: Span, needle: &str) -> BytePos;
    fn span_before(&self, original: Span, needle: &str) -> BytePos;
}

impl SpanUtils for CodeMap {
    #[inline]
    fn span_after(&self, original: Span, needle: &str) -> BytePos {
        let snippet = self.span_to_snippet(original).unwrap();
        let offset = snippet.find_uncommented(needle).unwrap() + needle.len();

        original.lo + BytePos(offset as u32)
    }

    #[inline]
    fn span_after_last(&self, original: Span, needle: &str) -> BytePos {
        let snippet = self.span_to_snippet(original).unwrap();
        let mut offset = 0;

        while let Some(additional_offset) = snippet[offset..].find_uncommented(needle) {
            offset += additional_offset + needle.len();
        }

        original.lo + BytePos(offset as u32)
    }

    #[inline]
    fn span_before(&self, original: Span, needle: &str) -> BytePos {
        let snippet = self.span_to_snippet(original).unwrap();
        let offset = snippet.find_uncommented(needle).unwrap();

        original.lo + BytePos(offset as u32)
    }
}
