use superslice::Ext;
use ::TextUnit;

#[derive(Clone, Debug)]
pub struct LineIndex {
    newlines: Vec<TextUnit>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LineCol {
    pub line: u32,
    pub col: TextUnit,
}

impl LineIndex {
    pub fn new(text: &str) -> LineIndex {
        let mut newlines = vec![0.into()];
        let mut curr = 0.into();
        for c in text.chars() {
            curr += TextUnit::of_char(c);
            if c == '\n' {
                newlines.push(curr);
            }
        }
        LineIndex { newlines }
    }

    pub fn line_col(&self, offset: TextUnit) -> LineCol {
        let line = self.newlines.upper_bound(&offset) - 1;
        let line_start_offset = self.newlines[line];
        let col = offset - line_start_offset;
        return LineCol { line: line as u32, col };
    }

    pub fn offset(&self, line_col: LineCol) -> TextUnit {
        //TODO: return Result
        self.newlines[line_col.line as usize] + line_col.col
    }
}

#[test]
fn test_line_index() {
    let text = "hello\nworld";
    let index = LineIndex::new(text);
    assert_eq!(index.line_col(0.into()), LineCol { line: 0, col: 0.into() });
    assert_eq!(index.line_col(1.into()), LineCol { line: 0, col: 1.into() });
    assert_eq!(index.line_col(5.into()), LineCol { line: 0, col: 5.into() });
    assert_eq!(index.line_col(6.into()), LineCol { line: 1, col: 0.into() });
    assert_eq!(index.line_col(7.into()), LineCol { line: 1, col: 1.into() });
    assert_eq!(index.line_col(8.into()), LineCol { line: 1, col: 2.into() });
    assert_eq!(index.line_col(10.into()), LineCol { line: 1, col: 4.into() });
    assert_eq!(index.line_col(11.into()), LineCol { line: 1, col: 5.into() });
    assert_eq!(index.line_col(12.into()), LineCol { line: 1, col: 6.into() });

    let text = "\nhello\nworld";
    let index = LineIndex::new(text);
    assert_eq!(index.line_col(0.into()), LineCol { line: 0, col: 0.into() });
    assert_eq!(index.line_col(1.into()), LineCol { line: 1, col: 0.into() });
    assert_eq!(index.line_col(2.into()), LineCol { line: 1, col: 1.into() });
    assert_eq!(index.line_col(6.into()), LineCol { line: 1, col: 5.into() });
    assert_eq!(index.line_col(7.into()), LineCol { line: 2, col: 0.into() });
}
