use super::*;

#[test]
fn slice_debug_output() {
    let input = unsafe { Slice::from_encoded_bytes_unchecked(b"\xF0hello,\tworld") };
    let expected = r#""\xF0hello,\tworld""#;
    let output = format!("{input:?}");

    assert_eq!(output, expected);
}

#[test]
fn display() {
    assert_eq!("Hello\u{FFFD}\u{FFFD} There\u{FFFD} Goodbye", unsafe {
        Slice::from_encoded_bytes_unchecked(b"Hello\xC0\x80 There\xE6\x83 Goodbye").to_string()
    },);
}
