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

#[test]
fn slice_starts_with_str() {
    let mut string = Buf::from_string(String::from("héllô="));
    string.push_slice(Slice::from_u8_slice(b"\xFF"));
    string.push_slice(Slice::from_str("wørld"));
    let slice = string.as_slice();

    assert!(slice.starts_with_str("héllô"));
    assert!(!slice.starts_with_str("héllô=wørld"));
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

#[test]
fn slice_strip_prefix_str() {
    let mut string = Buf::from_string(String::from("héllô="));
    string.push_slice(Slice::from_u8_slice(b"\xFF"));
    string.push_slice(Slice::from_str("wørld"));
    let slice = string.as_slice();

    assert!(slice.strip_prefix_str("héllô=wørld").is_none());

    let suffix = slice.strip_prefix_str("héllô");
    assert!(suffix.is_some());
    assert_eq!(&suffix.unwrap().inner, b"=\xFFw\xC3\xB8rld");
}

#[test]
fn slice_split_once() {
    let mut string = Buf::from_string(String::from("héllô="));
    string.push_slice(Slice::from_u8_slice(b"\xFF"));
    string.push_slice(Slice::from_str("wørld"));
    let slice = string.as_slice();

    let split = slice.split_once('=');
    assert!(split.is_some());
    let (prefix, suffix) = split.unwrap();
    assert_eq!(prefix, "héllô");
    assert_eq!(&suffix.inner, b"\xFFw\xC3\xB8rld");
}
