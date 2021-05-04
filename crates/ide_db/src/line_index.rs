//! `LineIndex` maps flat `TextSize` offsets into `(Line, Column)`
//! representation.
use std::iter;

use rustc_hash::FxHashMap;
use syntax::{TextRange, TextSize};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LineIndex {
    /// Offset the the beginning of each line, zero-based
    pub(crate) newlines: Vec<TextSize>,
    /// List of non-ASCII characters on each line
    pub(crate) utf16_lines: FxHashMap<u32, Vec<Utf16Char>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LineColUtf16 {
    /// Zero-based
    pub line: u32,
    /// Zero-based
    pub col: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LineCol {
    /// Zero-based
    pub line: u32,
    /// Zero-based utf8 offset
    pub col: u32,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub(crate) struct Utf16Char {
    /// Start offset of a character inside a line, zero-based
    pub(crate) start: TextSize,
    /// End offset of a character inside a line, zero-based
    pub(crate) end: TextSize,
}

impl Utf16Char {
    /// Returns the length in 8-bit UTF-8 code units.
    fn len(&self) -> TextSize {
        self.end - self.start
    }

    /// Returns the length in 16-bit UTF-16 code units.
    fn len_utf16(&self) -> usize {
        if self.len() == TextSize::from(4) {
            2
        } else {
            1
        }
    }
}

impl LineIndex {
    pub fn new(text: &str) -> LineIndex {
        let mut utf16_lines = FxHashMap::default();
        let mut utf16_chars = Vec::new();

        let mut newlines = vec![0.into()];
        let mut curr_row = 0.into();
        let mut curr_col = 0.into();
        let mut line = 0;
        for c in text.chars() {
            let c_len = TextSize::of(c);
            curr_row += c_len;
            if c == '\n' {
                newlines.push(curr_row);

                // Save any utf-16 characters seen in the previous line
                if !utf16_chars.is_empty() {
                    utf16_lines.insert(line, utf16_chars);
                    utf16_chars = Vec::new();
                }

                // Prepare for processing the next line
                curr_col = 0.into();
                line += 1;
                continue;
            }

            if !c.is_ascii() {
                utf16_chars.push(Utf16Char { start: curr_col, end: curr_col + c_len });
            }

            curr_col += c_len;
        }

        // Save any utf-16 characters seen in the last line
        if !utf16_chars.is_empty() {
            utf16_lines.insert(line, utf16_chars);
        }

        LineIndex { newlines, utf16_lines }
    }

    pub fn line_col(&self, offset: TextSize) -> LineCol {
        let line = self.newlines.partition_point(|&it| it <= offset) - 1;
        let line_start_offset = self.newlines[line];
        let col = offset - line_start_offset;
        LineCol { line: line as u32, col: col.into() }
    }

    pub fn offset(&self, line_col: LineCol) -> TextSize {
        self.newlines[line_col.line as usize] + TextSize::from(line_col.col)
    }

    pub fn to_utf16(&self, line_col: LineCol) -> LineColUtf16 {
        let col = self.utf8_to_utf16_col(line_col.line, line_col.col.into());
        LineColUtf16 { line: line_col.line, col: col as u32 }
    }

    pub fn to_utf8(&self, line_col: LineColUtf16) -> LineCol {
        let col = self.utf16_to_utf8_col(line_col.line, line_col.col);
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

    fn utf8_to_utf16_col(&self, line: u32, col: TextSize) -> usize {
        let mut res: usize = col.into();
        if let Some(utf16_chars) = self.utf16_lines.get(&line) {
            for c in utf16_chars {
                if c.end <= col {
                    res -= usize::from(c.len()) - c.len_utf16();
                } else {
                    // From here on, all utf16 characters come *after* the character we are mapping,
                    // so we don't need to take them into account
                    break;
                }
            }
        }
        res
    }

    fn utf16_to_utf8_col(&self, line: u32, mut col: u32) -> TextSize {
        if let Some(utf16_chars) = self.utf16_lines.get(&line) {
            for c in utf16_chars {
                if col > u32::from(c.start) {
                    col += u32::from(c.len()) - c.len_utf16() as u32;
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
mod tests;
