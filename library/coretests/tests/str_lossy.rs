#[test]
fn chunks() {
    macro_rules! assert_chunks {
        ( $string:expr, $(($valid:expr, $invalid:expr)),* $(,)? ) => {{
            let mut iter = $string.utf8_chunks();
            $(
                let chunk = iter.next().expect("missing chunk");
                assert_eq!($valid, chunk.valid());
                assert_eq!($invalid, chunk.invalid());
            )*
            assert_eq!(None, iter.next());
        }};
    }

    assert_chunks!(b"hello", ("hello", b""));
    assert_chunks!("ศไทย中华Việt Nam".as_bytes(), ("ศไทย中华Việt Nam", b""));
    assert_chunks!(
        b"Hello\xC2 There\xFF Goodbye",
        ("Hello", b"\xC2"),
        (" There", b"\xFF"),
        (" Goodbye", b""),
    );
    assert_chunks!(
        b"Hello\xC0\x80 There\xE6\x83 Goodbye",
        ("Hello", b"\xC0"),
        ("", b"\x80"),
        (" There", b"\xE6\x83"),
        (" Goodbye", b""),
    );
    assert_chunks!(
        b"\xF5foo\xF5\x80bar",
        ("", b"\xF5"),
        ("foo", b"\xF5"),
        ("", b"\x80"),
        ("bar", b""),
    );
    assert_chunks!(
        b"\xF1foo\xF1\x80bar\xF1\x80\x80baz",
        ("", b"\xF1"),
        ("foo", b"\xF1\x80"),
        ("bar", b"\xF1\x80\x80"),
        ("baz", b""),
    );
    assert_chunks!(
        b"\xF4foo\xF4\x80bar\xF4\xBFbaz",
        ("", b"\xF4"),
        ("foo", b"\xF4\x80"),
        ("bar", b"\xF4"),
        ("", b"\xBF"),
        ("baz", b""),
    );
    assert_chunks!(
        b"\xF0\x80\x80\x80foo\xF0\x90\x80\x80bar",
        ("", b"\xF0"),
        ("", b"\x80"),
        ("", b"\x80"),
        ("", b"\x80"),
        ("foo\u{10000}bar", b""),
    );

    // surrogates
    assert_chunks!(
        b"\xED\xA0\x80foo\xED\xBF\xBFbar",
        ("", b"\xED"),
        ("", b"\xA0"),
        ("", b"\x80"),
        ("foo", b"\xED"),
        ("", b"\xBF"),
        ("", b"\xBF"),
        ("bar", b""),
    );
}

#[test]
fn debug() {
    assert_eq!(
        "\"Hello\\xC0\\x80 There\\xE6\\x83 Goodbye\\u{10d4ea}\"",
        &format!(
            "{:?}",
            b"Hello\xC0\x80 There\xE6\x83 Goodbye\xf4\x8d\x93\xaa".utf8_chunks().debug(),
        ),
    );
}
