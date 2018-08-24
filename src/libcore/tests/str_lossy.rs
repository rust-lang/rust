use core::str::lossy::*;

#[test]
fn chunks() {
    let mut iter = Utf8Lossy::from_bytes(b"hello").chunks();
    assert_eq!(Some(Utf8LossyChunk { valid: "hello", broken: b"", }), iter.next());
    assert_eq!(None, iter.next());

    let mut iter = Utf8Lossy::from_bytes("ศไทย中华Việt Nam".as_bytes()).chunks();
    assert_eq!(Some(Utf8LossyChunk { valid: "ศไทย中华Việt Nam", broken: b"", }), iter.next());
    assert_eq!(None, iter.next());

    let mut iter = Utf8Lossy::from_bytes(b"Hello\xC2 There\xFF Goodbye").chunks();
    assert_eq!(Some(Utf8LossyChunk { valid: "Hello", broken: b"\xC2", }), iter.next());
    assert_eq!(Some(Utf8LossyChunk { valid: " There", broken: b"\xFF", }), iter.next());
    assert_eq!(Some(Utf8LossyChunk { valid: " Goodbye", broken: b"", }), iter.next());
    assert_eq!(None, iter.next());

    let mut iter = Utf8Lossy::from_bytes(b"Hello\xC0\x80 There\xE6\x83 Goodbye").chunks();
    assert_eq!(Some(Utf8LossyChunk { valid: "Hello", broken: b"\xC0", }), iter.next());
    assert_eq!(Some(Utf8LossyChunk { valid: "", broken: b"\x80", }), iter.next());
    assert_eq!(Some(Utf8LossyChunk { valid: " There", broken: b"\xE6\x83", }), iter.next());
    assert_eq!(Some(Utf8LossyChunk { valid: " Goodbye", broken: b"", }), iter.next());
    assert_eq!(None, iter.next());

    let mut iter = Utf8Lossy::from_bytes(b"\xF5foo\xF5\x80bar").chunks();
    assert_eq!(Some(Utf8LossyChunk { valid: "", broken: b"\xF5", }), iter.next());
    assert_eq!(Some(Utf8LossyChunk { valid: "foo", broken: b"\xF5", }), iter.next());
    assert_eq!(Some(Utf8LossyChunk { valid: "", broken: b"\x80", }), iter.next());
    assert_eq!(Some(Utf8LossyChunk { valid: "bar", broken: b"", }), iter.next());
    assert_eq!(None, iter.next());

    let mut iter = Utf8Lossy::from_bytes(b"\xF1foo\xF1\x80bar\xF1\x80\x80baz").chunks();
    assert_eq!(Some(Utf8LossyChunk { valid: "", broken: b"\xF1", }), iter.next());
    assert_eq!(Some(Utf8LossyChunk { valid: "foo", broken: b"\xF1\x80", }), iter.next());
    assert_eq!(Some(Utf8LossyChunk { valid: "bar", broken: b"\xF1\x80\x80", }), iter.next());
    assert_eq!(Some(Utf8LossyChunk { valid: "baz", broken: b"", }), iter.next());
    assert_eq!(None, iter.next());

    let mut iter = Utf8Lossy::from_bytes(b"\xF4foo\xF4\x80bar\xF4\xBFbaz").chunks();
    assert_eq!(Some(Utf8LossyChunk { valid: "", broken: b"\xF4", }), iter.next());
    assert_eq!(Some(Utf8LossyChunk { valid: "foo", broken: b"\xF4\x80", }), iter.next());
    assert_eq!(Some(Utf8LossyChunk { valid: "bar", broken: b"\xF4", }), iter.next());
    assert_eq!(Some(Utf8LossyChunk { valid: "", broken: b"\xBF", }), iter.next());
    assert_eq!(Some(Utf8LossyChunk { valid: "baz", broken: b"", }), iter.next());
    assert_eq!(None, iter.next());

    let mut iter = Utf8Lossy::from_bytes(b"\xF0\x80\x80\x80foo\xF0\x90\x80\x80bar").chunks();
    assert_eq!(Some(Utf8LossyChunk { valid: "", broken: b"\xF0", }), iter.next());
    assert_eq!(Some(Utf8LossyChunk { valid: "", broken: b"\x80", }), iter.next());
    assert_eq!(Some(Utf8LossyChunk { valid: "", broken: b"\x80", }), iter.next());
    assert_eq!(Some(Utf8LossyChunk { valid: "", broken: b"\x80", }), iter.next());
    assert_eq!(Some(Utf8LossyChunk { valid: "foo\u{10000}bar", broken: b"", }), iter.next());
    assert_eq!(None, iter.next());

    // surrogates
    let mut iter = Utf8Lossy::from_bytes(b"\xED\xA0\x80foo\xED\xBF\xBFbar").chunks();
    assert_eq!(Some(Utf8LossyChunk { valid: "", broken: b"\xED", }), iter.next());
    assert_eq!(Some(Utf8LossyChunk { valid: "", broken: b"\xA0", }), iter.next());
    assert_eq!(Some(Utf8LossyChunk { valid: "", broken: b"\x80", }), iter.next());
    assert_eq!(Some(Utf8LossyChunk { valid: "foo", broken: b"\xED", }), iter.next());
    assert_eq!(Some(Utf8LossyChunk { valid: "", broken: b"\xBF", }), iter.next());
    assert_eq!(Some(Utf8LossyChunk { valid: "", broken: b"\xBF", }), iter.next());
    assert_eq!(Some(Utf8LossyChunk { valid: "bar", broken: b"", }), iter.next());
    assert_eq!(None, iter.next());
}

#[test]
fn display() {
    assert_eq!(
        "Hello\u{FFFD}\u{FFFD} There\u{FFFD} Goodbye",
        &Utf8Lossy::from_bytes(b"Hello\xC0\x80 There\xE6\x83 Goodbye").to_string());
}

#[test]
fn debug() {
    assert_eq!(
        "\"Hello\\xc0\\x80 There\\xe6\\x83 Goodbye\\u{10d4ea}\"",
        &format!("{:?}", Utf8Lossy::from_bytes(
            b"Hello\xC0\x80 There\xE6\x83 Goodbye\xf4\x8d\x93\xaa")));
}
