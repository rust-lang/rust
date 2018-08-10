use superslice::Ext;
use ::{TextUnit};

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

    pub fn translate(&self, offset: TextUnit) -> LineCol {
        let line = self.newlines.upper_bound(&offset) - 1;
        let line_start_offset = self.newlines[line];
        let col = offset - line_start_offset;
        return LineCol { line: line as u32, col }
    }
}

#[test]
fn test_line_index() {
    let text = "hello\nworld";
    let index = LineIndex::new(text);
    assert_eq!(index.translate(0.into()), LineCol { line: 0, col: 0.into()});
    assert_eq!(index.translate(1.into()), LineCol { line: 0, col: 1.into()});
    assert_eq!(index.translate(5.into()), LineCol { line: 0, col: 5.into()});
    assert_eq!(index.translate(6.into()), LineCol { line: 1, col: 0.into()});
    assert_eq!(index.translate(7.into()), LineCol { line: 1, col: 1.into()});
    assert_eq!(index.translate(8.into()), LineCol { line: 1, col: 2.into()});
    assert_eq!(index.translate(10.into()), LineCol { line: 1, col: 4.into()});
    assert_eq!(index.translate(11.into()), LineCol { line: 1, col: 5.into()});
    assert_eq!(index.translate(12.into()), LineCol { line: 1, col: 6.into()});

    let text = "\nhello\nworld";
    let index = LineIndex::new(text);
    assert_eq!(index.translate(0.into()), LineCol { line: 0, col: 0.into()});
    assert_eq!(index.translate(1.into()), LineCol { line: 1, col: 0.into()});
    assert_eq!(index.translate(2.into()), LineCol { line: 1, col: 1.into()});
    assert_eq!(index.translate(6.into()), LineCol { line: 1, col: 5.into()});
    assert_eq!(index.translate(7.into()), LineCol { line: 2, col: 0.into()});
}
