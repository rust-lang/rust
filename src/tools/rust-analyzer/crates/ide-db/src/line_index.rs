//! `LineIndex` maps flat `TextSize` offsets into `(Line, Column)`
//! representation.
use std::{iter, mem};

use stdx::hash::NoHashHashMap;
use syntax::{TextRange, TextSize};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LineIndex {
    /// Offset the the beginning of each line, zero-based
    pub(crate) newlines: Vec<TextSize>,
    /// List of non-ASCII characters on each line
    pub(crate) utf16_lines: NoHashHashMap<u32, Vec<Utf16Char>>,
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
        let mut utf16_lines = NoHashHashMap::default();
        let mut utf16_chars = Vec::new();

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
                if !utf16_chars.is_empty() {
                    utf16_lines.insert(line, mem::take(&mut utf16_chars));
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

    pub fn offset(&self, line_col: LineCol) -> Option<TextSize> {
        self.newlines
            .get(line_col.line as usize)
            .map(|offset| offset + TextSize::from(line_col.col))
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
mod tests {
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
        assert_eq!(col_index.utf16_lines.len(), 0);
    }

    #[test]
    fn test_single_char() {
        let col_index = LineIndex::new(
            "
const C: char = 'メ';
",
        );

        assert_eq!(col_index.utf16_lines.len(), 1);
        assert_eq!(col_index.utf16_lines[&1].len(), 1);
        assert_eq!(col_index.utf16_lines[&1][0], Utf16Char { start: 17.into(), end: 20.into() });

        // UTF-8 to UTF-16, no changes
        assert_eq!(col_index.utf8_to_utf16_col(1, 15.into()), 15);

        // UTF-8 to UTF-16
        assert_eq!(col_index.utf8_to_utf16_col(1, 22.into()), 20);

        // UTF-16 to UTF-8, no changes
        assert_eq!(col_index.utf16_to_utf8_col(1, 15), TextSize::from(15));

        // UTF-16 to UTF-8
        assert_eq!(col_index.utf16_to_utf8_col(1, 19), TextSize::from(21));

        let col_index = LineIndex::new("a𐐏b");
        assert_eq!(col_index.utf16_to_utf8_col(0, 3), TextSize::from(5));
    }

    #[test]
    fn test_string() {
        let col_index = LineIndex::new(
            "
const C: char = \"メ メ\";
",
        );

        assert_eq!(col_index.utf16_lines.len(), 1);
        assert_eq!(col_index.utf16_lines[&1].len(), 2);
        assert_eq!(col_index.utf16_lines[&1][0], Utf16Char { start: 17.into(), end: 20.into() });
        assert_eq!(col_index.utf16_lines[&1][1], Utf16Char { start: 21.into(), end: 24.into() });

        // UTF-8 to UTF-16
        assert_eq!(col_index.utf8_to_utf16_col(1, 15.into()), 15);

        assert_eq!(col_index.utf8_to_utf16_col(1, 21.into()), 19);
        assert_eq!(col_index.utf8_to_utf16_col(1, 25.into()), 21);

        assert!(col_index.utf8_to_utf16_col(2, 15.into()) == 15);

        // UTF-16 to UTF-8
        assert_eq!(col_index.utf16_to_utf8_col(1, 15), TextSize::from(15));

        // メ UTF-8: 0xE3 0x83 0xA1, UTF-16: 0x30E1
        assert_eq!(col_index.utf16_to_utf8_col(1, 17), TextSize::from(17)); // first メ at 17..20
        assert_eq!(col_index.utf16_to_utf8_col(1, 18), TextSize::from(20)); // space
        assert_eq!(col_index.utf16_to_utf8_col(1, 19), TextSize::from(21)); // second メ at 21..24

        assert_eq!(col_index.utf16_to_utf8_col(2, 15), TextSize::from(15));
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
