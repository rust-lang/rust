//! See [`LineIndex`].

#![deny(missing_debug_implementations, missing_docs, rust_2018_idioms)]

#[cfg(test)]
mod tests;

use nohash_hasher::IntMap;

pub use text_size::{TextRange, TextSize};

/// Line/Column information in native, utf8 format.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LineCol {
    /// Zero-based.
    pub line: u32,
    /// Zero-based UTF-8 offset.
    pub col: u32,
}

/// A kind of wide character encoding.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum WideEncoding {
    /// UTF-16.
    Utf16,
    /// UTF-32.
    Utf32,
}

impl WideEncoding {
    /// Returns the number of units it takes to encode `text` in this encoding.
    pub fn measure(&self, text: &str) -> usize {
        match self {
            WideEncoding::Utf16 => text.encode_utf16().count(),
            WideEncoding::Utf32 => text.chars().count(),
        }
    }
}

/// Line/Column information in legacy encodings.
//
// Deliberately not a generic type and different from `LineCol`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WideLineCol {
    /// Zero-based.
    pub line: u32,
    /// Zero-based.
    pub col: u32,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct WideChar {
    /// Start offset of a character inside a line, zero-based
    start: TextSize,
    /// End offset of a character inside a line, zero-based
    end: TextSize,
}

impl WideChar {
    /// Returns the length in 8-bit UTF-8 code units.
    fn len(&self) -> TextSize {
        self.end - self.start
    }

    /// Returns the length in UTF-16 or UTF-32 code units.
    fn wide_len(&self, enc: WideEncoding) -> usize {
        match enc {
            WideEncoding::Utf16 => {
                if self.len() == TextSize::from(4) {
                    2
                } else {
                    1
                }
            }

            WideEncoding::Utf32 => 1,
        }
    }
}

/// Maps flat [`TextSize`] offsets into `(line, column)` representation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LineIndex {
    /// Offset the beginning of each line, zero-based.
    newlines: Box<[TextSize]>,
    /// List of non-ASCII characters on each line.
    line_wide_chars: IntMap<u32, Box<[WideChar]>>,
}

impl LineIndex {
    /// Returns a `LineIndex` for the `text`.
    pub fn new(text: &str) -> LineIndex {
        let mut line_wide_chars = IntMap::default();
        let mut wide_chars = Vec::new();

        let mut newlines = Vec::with_capacity(16);
        newlines.push(TextSize::from(0));

        let mut cur_row = 0.into();
        let mut cur_col = 0.into();
        let mut line = 0;
        for c in text.chars() {
            let c_len = TextSize::of(c);
            cur_row += c_len;
            if c == '\n' {
                newlines.push(cur_row);

                // Save any utf-16 characters seen in the previous line
                if !wide_chars.is_empty() {
                    line_wide_chars
                        .insert(line, std::mem::take(&mut wide_chars).into_boxed_slice());
                }

                // Prepare for processing the next line
                cur_col = 0.into();
                line += 1;
                continue;
            }

            if !c.is_ascii() {
                wide_chars.push(WideChar { start: cur_col, end: cur_col + c_len });
            }

            cur_col += c_len;
        }

        // Save any utf-16 characters seen in the last line
        if !wide_chars.is_empty() {
            line_wide_chars.insert(line, wide_chars.into_boxed_slice());
        }

        LineIndex { newlines: newlines.into_boxed_slice(), line_wide_chars }
    }

    /// Transforms the `TextSize` into a `LineCol`.
    pub fn line_col(&self, offset: TextSize) -> LineCol {
        let line = self.newlines.partition_point(|&it| it <= offset) - 1;
        let line_start_offset = self.newlines[line];
        let col = offset - line_start_offset;
        LineCol { line: line as u32, col: col.into() }
    }

    /// Transforms the `LineCol` into a `TextSize`.
    pub fn offset(&self, line_col: LineCol) -> Option<TextSize> {
        self.newlines
            .get(line_col.line as usize)
            .map(|offset| offset + TextSize::from(line_col.col))
    }

    /// Transforms the `LineCol` with the given `WideEncoding` into a `WideLineCol`.
    pub fn to_wide(&self, enc: WideEncoding, line_col: LineCol) -> WideLineCol {
        let col = self.utf8_to_wide_col(enc, line_col.line, line_col.col.into());
        WideLineCol { line: line_col.line, col: col as u32 }
    }

    /// Transforms the `WideLineCol` with the given `WideEncoding` into a `LineCol`.
    pub fn to_utf8(&self, enc: WideEncoding, line_col: WideLineCol) -> LineCol {
        let col = self.wide_to_utf8_col(enc, line_col.line, line_col.col);
        LineCol { line: line_col.line, col: col.into() }
    }

    /// Returns an iterator over the ranges for the lines.
    pub fn lines(&self, range: TextRange) -> impl Iterator<Item = TextRange> + '_ {
        let lo = self.newlines.partition_point(|&it| it < range.start());
        let hi = self.newlines.partition_point(|&it| it <= range.end());
        let all = std::iter::once(range.start())
            .chain(self.newlines[lo..hi].iter().copied())
            .chain(std::iter::once(range.end()));

        all.clone()
            .zip(all.skip(1))
            .map(|(lo, hi)| TextRange::new(lo, hi))
            .filter(|it| !it.is_empty())
    }

    fn utf8_to_wide_col(&self, enc: WideEncoding, line: u32, col: TextSize) -> usize {
        let mut res: usize = col.into();
        if let Some(wide_chars) = self.line_wide_chars.get(&line) {
            for c in wide_chars.iter() {
                if c.end <= col {
                    res -= usize::from(c.len()) - c.wide_len(enc);
                } else {
                    // From here on, all utf16 characters come *after* the character we are mapping,
                    // so we don't need to take them into account
                    break;
                }
            }
        }
        res
    }

    fn wide_to_utf8_col(&self, enc: WideEncoding, line: u32, mut col: u32) -> TextSize {
        if let Some(wide_chars) = self.line_wide_chars.get(&line) {
            for c in wide_chars.iter() {
                if col > u32::from(c.start) {
                    col += u32::from(c.len()) - c.wide_len(enc) as u32;
                } else {
                    // From here on, all utf16 characters come *after* the character we are mapping,
                    // so we don't need to take them into account
                    break;
                }
            }
        }

        col.into()
    }
}
