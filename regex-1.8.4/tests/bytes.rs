// These are tests specifically crafted for regexes that can match arbitrary
// bytes.

// A silly wrapper to make it possible to write and match raw bytes.
struct R<'a>(&'a [u8]);
impl<'a> R<'a> {
    fn as_bytes(&self) -> &'a [u8] {
        self.0
    }
}

mat!(word_boundary, r"(?-u) \b", " δ", None);
#[cfg(feature = "unicode-perl")]
mat!(word_boundary_unicode, r" \b", " δ", Some((0, 1)));
mat!(word_not_boundary, r"(?-u) \B", " δ", Some((0, 1)));
#[cfg(feature = "unicode-perl")]
mat!(word_not_boundary_unicode, r" \B", " δ", None);

mat!(perl_w_ascii, r"(?-u)\w+", "aδ", Some((0, 1)));
#[cfg(feature = "unicode-perl")]
mat!(perl_w_unicode, r"\w+", "aδ", Some((0, 3)));
mat!(perl_d_ascii, r"(?-u)\d+", "1२३9", Some((0, 1)));
#[cfg(feature = "unicode-perl")]
mat!(perl_d_unicode, r"\d+", "1२३9", Some((0, 8)));
mat!(perl_s_ascii, r"(?-u)\s+", " \u{1680}", Some((0, 1)));
#[cfg(feature = "unicode-perl")]
mat!(perl_s_unicode, r"\s+", " \u{1680}", Some((0, 4)));

// The first `(.+)` matches two Unicode codepoints, but can't match the 5th
// byte, which isn't valid UTF-8. The second (byte based) `(.+)` takes over and
// matches.
mat!(
    mixed1,
    r"(.+)(?-u)(.+)",
    R(b"\xCE\x93\xCE\x94\xFF"),
    Some((0, 5)),
    Some((0, 4)),
    Some((4, 5))
);

mat!(case_ascii_one, r"(?i-u)a", "A", Some((0, 1)));
mat!(case_ascii_class, r"(?i-u)[a-z]+", "AaAaA", Some((0, 5)));
#[cfg(feature = "unicode-case")]
mat!(case_unicode, r"(?i)[a-z]+", "aA\u{212A}aA", Some((0, 7)));
mat!(case_not_unicode, r"(?i-u)[a-z]+", "aA\u{212A}aA", Some((0, 2)));

mat!(negate_unicode, r"[^a]", "δ", Some((0, 2)));
mat!(negate_not_unicode, r"(?-u)[^a]", "δ", Some((0, 1)));

// This doesn't match in a normal Unicode regex because the implicit preceding
// `.*?` is Unicode aware.
mat!(dotstar_prefix_not_unicode1, r"(?-u)a", R(b"\xFFa"), Some((1, 2)));
mat!(dotstar_prefix_not_unicode2, r"a", R(b"\xFFa"), Some((1, 2)));

// Have fun with null bytes.
mat!(
    null_bytes,
    r"(?-u)(?P<cstr>[^\x00]+)\x00",
    R(b"foo\x00"),
    Some((0, 4)),
    Some((0, 3))
);

// Test that lookahead operators work properly in the face of invalid UTF-8.
// See: https://github.com/rust-lang/regex/issues/277
matiter!(
    invalidutf8_anchor1,
    r"(?-u)\xcc?^",
    R(b"\x8d#;\x1a\xa4s3\x05foobarX\\\x0f0t\xe4\x9b\xa4"),
    (0, 0)
);
matiter!(
    invalidutf8_anchor2,
    r"(?-u)^\xf7|4\xff\d\x8a\x8a\x8a\x8a\x8a\x8a\x8a\x8a\x8a\x8a\x8a\x8a\x8a##########[] d\x8a\x8a\x8a\x8a\x8a\x8a\x8a\x8a\x8a\x8a\x8a\x8a\x8a##########\[] #####\x80\S7|$",
    R(b"\x8d#;\x1a\xa4s3\x05foobarX\\\x0f0t\xe4\x9b\xa4"),
    (22, 22)
);
matiter!(
    invalidutf8_anchor3,
    r"(?-u)^|ddp\xff\xffdddddlQd@\x80",
    R(b"\x8d#;\x1a\xa4s3\x05foobarX\\\x0f0t\xe4\x9b\xa4"),
    (0, 0)
);

// See https://github.com/rust-lang/regex/issues/303
#[test]
fn negated_full_byte_range() {
    assert!(::regex::bytes::Regex::new(r#"(?-u)[^\x00-\xff]"#).is_err());
}

matiter!(word_boundary_ascii1, r"(?-u:\B)x(?-u:\B)", "áxβ");
matiter!(
    word_boundary_ascii2,
    r"(?-u:\B)",
    "0\u{7EF5E}",
    (2, 2),
    (3, 3),
    (4, 4),
    (5, 5)
);

// See: https://github.com/rust-lang/regex/issues/264
mat!(ascii_boundary_no_capture, r"(?-u)\B", "\u{28f3e}", Some((0, 0)));
mat!(ascii_boundary_capture, r"(?-u)(\B)", "\u{28f3e}", Some((0, 0)));

// See: https://github.com/rust-lang/regex/issues/271
mat!(end_not_wb, r"$(?-u:\B)", "\u{5c124}\u{b576c}", Some((8, 8)));
