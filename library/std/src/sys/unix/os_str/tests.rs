use super::*;

#[test]
fn slice_debug_output() {
    let input = Slice::from_u8_slice(b"\xF0hello,\tworld");
    let expected = r#""\xF0hello,\tworld""#;
    let output = format!("{input:?}");

    assert_eq!(output, expected);
}

#[test]
fn display() {
    assert_eq!(
        "Hello\u{FFFD}\u{FFFD} There\u{FFFD} Goodbye",
        Slice::from_u8_slice(b"Hello\xC0\x80 There\xE6\x83 Goodbye").to_string(),
    );
}
