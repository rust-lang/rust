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

#[test]
fn slice_starts_with() {
    let mut string = Buf::from_string(String::from("héllô="));
    string.push_slice(Slice::from_u8_slice(b"\xFF"));
    string.push_slice(Slice::from_str("wørld"));
    let slice = string.as_slice();

    assert!(slice.starts_with('h'));
    assert!(slice.starts_with("héllô"));
    assert!(!slice.starts_with("héllô=wørld"));
}

#[test]
fn slice_strip_prefix() {
    let mut string = Buf::from_string(String::from("héllô="));
    string.push_slice(Slice::from_u8_slice(b"\xFF"));
    string.push_slice(Slice::from_str("wørld"));
    let slice = string.as_slice();

    assert!(slice.strip_prefix("héllô=wørld").is_none());

    {
        let suffix = slice.strip_prefix('h');
        assert!(suffix.is_some());
        assert_eq!(&suffix.unwrap().inner, b"\xC3\xA9ll\xC3\xB4=\xFFw\xC3\xB8rld",);
    }

    {
        let suffix = slice.strip_prefix("héllô");
        assert!(suffix.is_some());
        assert_eq!(&suffix.unwrap().inner, b"=\xFFw\xC3\xB8rld");
    }
}
