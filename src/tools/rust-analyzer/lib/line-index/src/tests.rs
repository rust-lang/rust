use super::LineIndex;

#[test]
fn test_empty_index() {
    let col_index = LineIndex::new(
        "
const C: char = 'x';
",
    );
    assert_eq!(col_index.line_wide_chars.len(), 0);
}
