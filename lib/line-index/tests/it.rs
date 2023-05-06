use line_index::{LineCol, LineIndex, TextRange};

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
