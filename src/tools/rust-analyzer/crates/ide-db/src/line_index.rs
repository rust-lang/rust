//! `LineIndex` maps flat `TextSize` offsets into `(Line, Column)`
//! representation.
use std::{iter, mem};

use stdx::hash::NoHashHashMap;
use syntax::{TextRange, TextSize};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LineIndex {
    /// Offset the beginning of each line, zero-based.
    pub(crate) newlines: Vec<TextSize>,
    /// List of non-ASCII characters on each line.
    pub(crate) line_wide_chars: NoHashHashMap<u32, Vec<WideChar>>,
}

/// Line/Column information in native, utf8 format.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LineCol {
    /// Zero-based
    pub line: u32,
    /// Zero-based utf8 offset
    pub col: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WideEncoding {
    Utf16,
    Utf32,
}

/// Line/Column information in legacy encodings.
///
/// Deliberately not a generic type and different from `LineCol`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WideLineCol {
    /// Zero-based
    pub line: u32,
    /// Zero-based
    pub col: u32,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub(crate) struct WideChar {
    /// Start offset of a character inside a line, zero-based
    pub(crate) start: TextSize,
    /// End offset of a character inside a line, zero-based
    pub(crate) end: TextSize,
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

impl LineIndex {
    pub fn new(text: &str) -> LineIndex {
        let mut line_wide_chars = NoHashHashMap::default();
        let mut wide_chars = Vec::new();

        let mut newlines = Vec::with_capacity(16);
        newlines.push(TextSize::from(0));

        let mut curr_row = 0.into();
        let mut curr_col = 0.into();
        let mut line = 0;
        for c in text.chars() {
            let c_len = TextSize::of(c);
            curr_row += c_len;
            if c == '\n' {
                newlines.push(curr_row);

                // Save any utf-16 characters seen in the previous line
                if !wide_chars.is_empty() {
                    line_wide_chars.insert(line, mem::take(&mut wide_chars));
                }

                // Prepare for processing the next line
                curr_col = 0.into();
                line += 1;
                continue;
            }

            if !c.is_ascii() {
                wide_chars.push(WideChar { start: curr_col, end: curr_col + c_len });
            }

            curr_col += c_len;
        }

        // Save any utf-16 characters seen in the last line
        if !wide_chars.is_empty() {
            line_wide_chars.insert(line, wide_chars);
        }

        LineIndex { newlines, line_wide_chars }
    }

    pub fn line_col(&self, offset: TextSize) -> LineCol {
        let line = self.newlines.partition_point(|&it| it <= offset) - 1;
        let line_start_offset = self.newlines[line];
        let col = offset - line_start_offset;
        LineCol { line: line as u32, col: col.into() }
    }

    pub fn offset(&self, line_col: LineCol) -> Option<TextSize> {
        self.newlines
            .get(line_col.line as usize)
            .map(|offset| offset + TextSize::from(line_col.col))
    }

    pub fn to_wide(&self, enc: WideEncoding, line_col: LineCol) -> WideLineCol {
        let col = self.utf8_to_wide_col(enc, line_col.line, line_col.col.into());
        WideLineCol { line: line_col.line, col: col as u32 }
    }

    pub fn to_utf8(&self, enc: WideEncoding, line_col: WideLineCol) -> LineCol {
        let col = self.wide_to_utf8_col(enc, line_col.line, line_col.col);
        LineCol { line: line_col.line, col: col.into() }
    }

    pub fn lines(&self, range: TextRange) -> impl Iterator<Item = TextRange> + '_ {
        let lo = self.newlines.partition_point(|&it| it < range.start());
        let hi = self.newlines.partition_point(|&it| it <= range.end());
        let all = iter::once(range.start())
            .chain(self.newlines[lo..hi].iter().copied())
            .chain(iter::once(range.end()));

        all.clone()
            .zip(all.skip(1))
            .map(|(lo, hi)| TextRange::new(lo, hi))
            .filter(|it| !it.is_empty())
    }

    fn utf8_to_wide_col(&self, enc: WideEncoding, line: u32, col: TextSize) -> usize {
        let mut res: usize = col.into();
        if let Some(wide_chars) = self.line_wide_chars.get(&line) {
            for c in wide_chars {
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
            for c in wide_chars {
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

#[cfg(test)]
mod tests {
    use test_utils::skip_slow_tests;

    use super::WideEncoding::{Utf16, Utf32};
    use super::*;

    #[test]
    fn test_line_index() {
        let text = "hello\nworld";
        let table = [
            (00, 0, 0),
            (01, 0, 1),
            (05, 0, 5),
            (06, 1, 0),
            (07, 1, 1),
            (08, 1, 2),
            (10, 1, 4),
            (11, 1, 5),
            (12, 1, 6),
        ];

        let index = LineIndex::new(text);
        for (offset, line, col) in table {
            assert_eq!(index.line_col(offset.into()), LineCol { line, col });
        }

        let text = "\nhello\nworld";
        let table = [(0, 0, 0), (1, 1, 0), (2, 1, 1), (6, 1, 5), (7, 2, 0)];
        let index = LineIndex::new(text);
        for (offset, line, col) in table {
            assert_eq!(index.line_col(offset.into()), LineCol { line, col });
        }
    }

    #[test]
    fn test_char_len() {
        assert_eq!('メ'.len_utf8(), 3);
        assert_eq!('メ'.len_utf16(), 1);
    }

    #[test]
    fn test_empty_index() {
        let col_index = LineIndex::new(
            "
const C: char = 'x';
",
        );
        assert_eq!(col_index.line_wide_chars.len(), 0);
    }

    #[test]
    fn test_every_chars() {
        if skip_slow_tests() {
            return;
        }

        let text: String = {
            let mut chars: Vec<char> = ((0 as char)..char::MAX).collect(); // Neat!
            chars.extend("\n".repeat(chars.len() / 16).chars());
            let mut rng = oorandom::Rand32::new(stdx::rand::seed());
            stdx::rand::shuffle(&mut chars, |i| rng.rand_range(0..i as u32) as usize);
            chars.into_iter().collect()
        };
        assert!(text.contains('💩')); // Sanity check.

        let line_index = LineIndex::new(&text);

        let mut lin_col = LineCol { line: 0, col: 0 };
        let mut col_utf16 = 0;
        let mut col_utf32 = 0;
        for (offset, c) in text.char_indices() {
            let got_offset = line_index.offset(lin_col).unwrap();
            assert_eq!(usize::from(got_offset), offset);

            let got_lin_col = line_index.line_col(got_offset);
            assert_eq!(got_lin_col, lin_col);

            for enc in [Utf16, Utf32] {
                let wide_lin_col = line_index.to_wide(enc, lin_col);
                let got_lin_col = line_index.to_utf8(enc, wide_lin_col);
                assert_eq!(got_lin_col, lin_col);

                let want_col = match enc {
                    Utf16 => col_utf16,
                    Utf32 => col_utf32,
                };
                assert_eq!(wide_lin_col.col, want_col)
            }

            if c == '\n' {
                lin_col.line += 1;
                lin_col.col = 0;
                col_utf16 = 0;
                col_utf32 = 0;
            } else {
                lin_col.col += c.len_utf8() as u32;
                col_utf16 += c.len_utf16() as u32;
                col_utf32 += 1;
            }
        }
    }

    #[test]
    fn test_splitlines() {
        fn r(lo: u32, hi: u32) -> TextRange {
            TextRange::new(lo.into(), hi.into())
        }

        let text = "a\nbb\nccc\n";
        let line_index = LineIndex::new(text);

        let actual = line_index.lines(r(0, 9)).collect::<Vec<_>>();
        let expected = vec![r(0, 2), r(2, 5), r(5, 9)];
        assert_eq!(actual, expected);

        let text = "";
        let line_index = LineIndex::new(text);

        let actual = line_index.lines(r(0, 0)).collect::<Vec<_>>();
        let expected = vec![];
        assert_eq!(actual, expected);

        let text = "\n";
        let line_index = LineIndex::new(text);

        let actual = line_index.lines(r(0, 1)).collect::<Vec<_>>();
        let expected = vec![r(0, 1)];
        assert_eq!(actual, expected)
    }
}
