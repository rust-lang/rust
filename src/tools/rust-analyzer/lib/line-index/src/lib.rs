//! See [`LineIndex`].

#![deny(missing_debug_implementations, missing_docs, rust_2018_idioms)]

#[cfg(test)]
mod tests;

use nohash_hasher::IntMap;

pub use text_size::{TextRange, TextSize};

/// `(line, column)` information in the native, UTF-8 encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LineCol {
    /// Zero-based.
    pub line: u32,
    /// Zero-based UTF-8 offset.
    pub col: u32,
}

/// A kind of wide character encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum WideEncoding {
    /// UTF-16.
    Utf16,
    /// UTF-32.
    Utf32,
}

impl WideEncoding {
    /// Returns the number of code units it takes to encode `text` in this encoding.
    pub fn measure(&self, text: &str) -> usize {
        match self {
            WideEncoding::Utf16 => text.encode_utf16().count(),
            WideEncoding::Utf32 => text.chars().count(),
        }
    }
}

/// `(line, column)` information in wide encodings.
///
/// See [`WideEncoding`] for the kinds of wide encodings available.
//
// Deliberately not a generic type and different from `LineCol`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WideLineCol {
    /// Zero-based.
    pub line: u32,
    /// Zero-based.
    pub col: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct WideChar {
    /// Start offset of a character inside a line, zero-based.
    start: TextSize,
    /// End offset of a character inside a line, zero-based.
    end: TextSize,
}

impl WideChar {
    /// Returns the length in 8-bit UTF-8 code units.
    fn len(&self) -> TextSize {
        self.end - self.start
    }

    /// Returns the length in UTF-16 or UTF-32 code units.
    fn wide_len(&self, enc: WideEncoding) -> u32 {
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

/// Maps flat [`TextSize`] offsets to/from `(line, column)` representation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LineIndex {
    /// Offset the beginning of each line (except the first, which always has offset 0).
    newlines: Box<[TextSize]>,
    /// List of non-ASCII characters on each line.
    line_wide_chars: IntMap<u32, Box<[WideChar]>>,
    /// The length of the entire text.
    len: TextSize,
}

impl LineIndex {
    /// Returns a `LineIndex` for the `text`.
    pub fn new(text: &str) -> LineIndex {
        let mut newlines = Vec::<TextSize>::with_capacity(16);
        let mut line_wide_chars = IntMap::<u32, Box<[WideChar]>>::default();

        let mut wide_chars = Vec::<WideChar>::new();
        let mut cur_row = TextSize::from(0);
        let mut cur_col = TextSize::from(0);
        let mut line = 0u32;

        for c in text.chars() {
            let c_len = TextSize::of(c);
            cur_row += c_len;
            if c == '\n' {
                newlines.push(cur_row);

                // Save any wide characters seen in the previous line
                if !wide_chars.is_empty() {
                    let cs = std::mem::take(&mut wide_chars).into_boxed_slice();
                    line_wide_chars.insert(line, cs);
                }

                // Prepare for processing the next line
                cur_col = TextSize::from(0);
                line += 1;
                continue;
            }

            if !c.is_ascii() {
                wide_chars.push(WideChar { start: cur_col, end: cur_col + c_len });
            }

            cur_col += c_len;
        }

        // Save any wide characters seen in the last line
        if !wide_chars.is_empty() {
            line_wide_chars.insert(line, wide_chars.into_boxed_slice());
        }

        LineIndex {
            newlines: newlines.into_boxed_slice(),
            line_wide_chars,
            len: TextSize::of(text),
        }
    }

    /// Transforms the `TextSize` into a `LineCol`.
    ///
    /// # Panics
    ///
    /// If the offset is invalid. See [`Self::try_line_col`].
    pub fn line_col(&self, offset: TextSize) -> LineCol {
        self.try_line_col(offset).expect("invalid offset")
    }

    /// Transforms the `TextSize` into a `LineCol`.
    ///
    /// Returns `None` if the `offset` was invalid, e.g. if it extends past the end of the text or
    /// points to the middle of a multi-byte character.
    pub fn try_line_col(&self, offset: TextSize) -> Option<LineCol> {
        if offset > self.len {
            return None;
        }
        let line = self.newlines.partition_point(|&it| it <= offset);
        let start = self.start_offset(line)?;
        let col = offset - start;
        let ret = LineCol { line: line as u32, col: col.into() };
        self.line_wide_chars
            .get(&ret.line)
            .into_iter()
            .flat_map(|it| it.iter())
            .all(|it| col <= it.start || it.end <= col)
            .then_some(ret)
    }

    /// Transforms the `LineCol` into a `TextSize`.
    pub fn offset(&self, line_col: LineCol) -> Option<TextSize> {
        self.start_offset(line_col.line as usize).map(|start| start + TextSize::from(line_col.col))
    }

    fn start_offset(&self, line: usize) -> Option<TextSize> {
        match line.checked_sub(1) {
            None => Some(TextSize::from(0)),
            Some(it) => self.newlines.get(it).copied(),
        }
    }

    /// Transforms the `LineCol` with the given `WideEncoding` into a `WideLineCol`.
    pub fn to_wide(&self, enc: WideEncoding, line_col: LineCol) -> Option<WideLineCol> {
        let mut col = line_col.col;
        if let Some(wide_chars) = self.line_wide_chars.get(&line_col.line) {
            for c in wide_chars.iter() {
                if u32::from(c.end) <= line_col.col {
                    col = col.checked_sub(u32::from(c.len()) - c.wide_len(enc))?;
                } else {
                    // From here on, all utf16 characters come *after* the character we are mapping,
                    // so we don't need to take them into account
                    break;
                }
            }
        }
        Some(WideLineCol { line: line_col.line, col })
    }

    /// Transforms the `WideLineCol` with the given `WideEncoding` into a `LineCol`.
    pub fn to_utf8(&self, enc: WideEncoding, line_col: WideLineCol) -> Option<LineCol> {
        let mut col = line_col.col;
        if let Some(wide_chars) = self.line_wide_chars.get(&line_col.line) {
            for c in wide_chars.iter() {
                if col > u32::from(c.start) {
                    col = col.checked_add(u32::from(c.len()) - c.wide_len(enc))?;
                } else {
                    // From here on, all utf16 characters come *after* the character we are mapping,
                    // so we don't need to take them into account
                    break;
                }
            }
        }
        Some(LineCol { line: line_col.line, col })
    }

    /// Given a range [start, end), returns a sorted iterator of non-empty ranges [start, x1), [x1,
    /// x2), ..., [xn, end) where all the xi, which are positions of newlines, are inside the range
    /// [start, end).
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

    /// Returns the length of the original text.
    pub fn len(&self) -> TextSize {
        self.len
    }
}
