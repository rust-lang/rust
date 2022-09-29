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
fn buf_into_string_split() {
    let mut string = Buf::from_string(String::from("héllô wørld"));
    {
        let (prefix, suffix) = string.clone().into_string_split();
        assert_eq!(prefix, String::from("héllô wørld"));
        assert_eq!(suffix.into_inner(), Vec::new());
    }

    string.push_slice(Slice::from_u8_slice(b"\xFF"));
    {
        let (prefix, suffix) = string.clone().into_string_split();
        assert_eq!(prefix, String::from("héllô wørld"));
        assert_eq!(suffix.into_inner(), vec![0xFF]);
    }
}

#[test]
fn slice_to_str_split() {
    let mut string = Buf::from_string(String::from("héllô wørld"));
    {
        let (prefix, suffix) = string.as_slice().to_str_split();
        assert_eq!(prefix, "héllô wørld");
        assert_eq!(&suffix.inner, b"");
    }

    string.push_slice(Slice::from_u8_slice(b"\xFF"));
    {
        let (prefix, suffix) = string.as_slice().to_str_split();
        assert_eq!(prefix, String::from("héllô wørld"));
        assert_eq!(&suffix.inner, b"\xFF");
    }
}
