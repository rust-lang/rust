use super::*;

#[test]
fn test_unescape_char_bad() {
    fn check(literal_text: &str, expected_error: EscapeError) {
        assert_eq!(unescape_char(literal_text), Err(expected_error));
    }

    check("", EscapeError::ZeroChars);
    check(r"\", EscapeError::LoneSlash);

    check("\n", EscapeError::EscapeOnlyChar);
    check("\t", EscapeError::EscapeOnlyChar);
    check("'", EscapeError::EscapeOnlyChar);
    check("\r", EscapeError::BareCarriageReturn);

    check("spam", EscapeError::MoreThanOneChar);
    check(r"\x0ff", EscapeError::MoreThanOneChar);
    check(r#"\"a"#, EscapeError::MoreThanOneChar);
    check(r"\na", EscapeError::MoreThanOneChar);
    check(r"\ra", EscapeError::MoreThanOneChar);
    check(r"\ta", EscapeError::MoreThanOneChar);
    check(r"\\a", EscapeError::MoreThanOneChar);
    check(r"\'a", EscapeError::MoreThanOneChar);
    check(r"\0a", EscapeError::MoreThanOneChar);
    check(r"\u{0}x", EscapeError::MoreThanOneChar);
    check(r"\u{1F63b}}", EscapeError::MoreThanOneChar);

    check(r"\v", EscapeError::InvalidEscape);
    check(r"\💩", EscapeError::InvalidEscape);
    check(r"\●", EscapeError::InvalidEscape);
    check("\\\r", EscapeError::InvalidEscape);

    check(r"\x", EscapeError::TooShortHexEscape);
    check(r"\x0", EscapeError::TooShortHexEscape);
    check(r"\xf", EscapeError::TooShortHexEscape);
    check(r"\xa", EscapeError::TooShortHexEscape);
    check(r"\xx", EscapeError::InvalidCharInHexEscape);
    check(r"\xы", EscapeError::InvalidCharInHexEscape);
    check(r"\x🦀", EscapeError::InvalidCharInHexEscape);
    check(r"\xtt", EscapeError::InvalidCharInHexEscape);
    check(r"\xff", EscapeError::OutOfRangeHexEscape);
    check(r"\xFF", EscapeError::OutOfRangeHexEscape);
    check(r"\x80", EscapeError::OutOfRangeHexEscape);

    check(r"\u", EscapeError::NoBraceInUnicodeEscape);
    check(r"\u[0123]", EscapeError::NoBraceInUnicodeEscape);
    check(r"\u{0x}", EscapeError::InvalidCharInUnicodeEscape);
    check(r"\u{", EscapeError::UnclosedUnicodeEscape);
    check(r"\u{0000", EscapeError::UnclosedUnicodeEscape);
    check(r"\u{}", EscapeError::EmptyUnicodeEscape);
    check(r"\u{_0000}", EscapeError::LeadingUnderscoreUnicodeEscape);
    check(r"\u{0000000}", EscapeError::OverlongUnicodeEscape);
    check(r"\u{FFFFFF}", EscapeError::OutOfRangeUnicodeEscape);
    check(r"\u{ffffff}", EscapeError::OutOfRangeUnicodeEscape);
    check(r"\u{ffffff}", EscapeError::OutOfRangeUnicodeEscape);

    check(r"\u{DC00}", EscapeError::LoneSurrogateUnicodeEscape);
    check(r"\u{DDDD}", EscapeError::LoneSurrogateUnicodeEscape);
    check(r"\u{DFFF}", EscapeError::LoneSurrogateUnicodeEscape);

    check(r"\u{D800}", EscapeError::LoneSurrogateUnicodeEscape);
    check(r"\u{DAAA}", EscapeError::LoneSurrogateUnicodeEscape);
    check(r"\u{DBFF}", EscapeError::LoneSurrogateUnicodeEscape);
}

#[test]
fn test_unescape_char_good() {
    fn check(literal_text: &str, expected_char: char) {
        assert_eq!(unescape_char(literal_text), Ok(expected_char));
    }

    check("a", 'a');
    check("ы", 'ы');
    check("🦀", '🦀');

    check(r#"\""#, '"');
    check(r"\n", '\n');
    check(r"\r", '\r');
    check(r"\t", '\t');
    check(r"\\", '\\');
    check(r"\'", '\'');
    check(r"\0", '\0');

    check(r"\x00", '\0');
    check(r"\x5a", 'Z');
    check(r"\x5A", 'Z');
    check(r"\x7f", 127 as char);

    check(r"\u{0}", '\0');
    check(r"\u{000000}", '\0');
    check(r"\u{41}", 'A');
    check(r"\u{0041}", 'A');
    check(r"\u{00_41}", 'A');
    check(r"\u{4__1__}", 'A');
    check(r"\u{1F63b}", '😻');
}

#[test]
fn test_unescape_str_warn() {
    fn check(literal: &str, expected: &[(Range<usize>, Result<char, EscapeError>)]) {
        let mut unescaped = Vec::with_capacity(literal.len());
        unescape_literal(literal, Mode::Str, &mut |range, res| unescaped.push((range, res)));
        assert_eq!(unescaped, expected);
    }

    // Check we can handle escaped newlines at the end of a file.
    check("\\\n", &[]);
    check("\\\n ", &[]);

    check(
        "\\\n \u{a0} x",
        &[
            (0..5, Err(EscapeError::UnskippedWhitespaceWarning)),
            (3..5, Ok('\u{a0}')),
            (5..6, Ok(' ')),
            (6..7, Ok('x')),
        ],
    );
    check("\\\n  \n  x", &[(0..7, Err(EscapeError::MultipleSkippedLinesWarning)), (7..8, Ok('x'))]);
}

#[test]
fn test_unescape_str_good() {
    fn check(literal_text: &str, expected: &str) {
        let mut buf = Ok(String::with_capacity(literal_text.len()));
        unescape_literal(literal_text, Mode::Str, &mut |range, c| {
            if let Ok(b) = &mut buf {
                match c {
                    Ok(c) => b.push(c),
                    Err(e) => buf = Err((range, e)),
                }
            }
        });
        assert_eq!(buf.as_deref(), Ok(expected))
    }

    check("foo", "foo");
    check("", "");
    check(" \t\n", " \t\n");

    check("hello \\\n     world", "hello world");
    check("thread's", "thread's")
}

#[test]
fn test_unescape_byte_bad() {
    fn check(literal_text: &str, expected_error: EscapeError) {
        assert_eq!(unescape_byte(literal_text), Err(expected_error));
    }

    check("", EscapeError::ZeroChars);
    check(r"\", EscapeError::LoneSlash);

    check("\n", EscapeError::EscapeOnlyChar);
    check("\t", EscapeError::EscapeOnlyChar);
    check("'", EscapeError::EscapeOnlyChar);
    check("\r", EscapeError::BareCarriageReturn);

    check("spam", EscapeError::MoreThanOneChar);
    check(r"\x0ff", EscapeError::MoreThanOneChar);
    check(r#"\"a"#, EscapeError::MoreThanOneChar);
    check(r"\na", EscapeError::MoreThanOneChar);
    check(r"\ra", EscapeError::MoreThanOneChar);
    check(r"\ta", EscapeError::MoreThanOneChar);
    check(r"\\a", EscapeError::MoreThanOneChar);
    check(r"\'a", EscapeError::MoreThanOneChar);
    check(r"\0a", EscapeError::MoreThanOneChar);

    check(r"\v", EscapeError::InvalidEscape);
    check(r"\💩", EscapeError::InvalidEscape);
    check(r"\●", EscapeError::InvalidEscape);

    check(r"\x", EscapeError::TooShortHexEscape);
    check(r"\x0", EscapeError::TooShortHexEscape);
    check(r"\xa", EscapeError::TooShortHexEscape);
    check(r"\xf", EscapeError::TooShortHexEscape);
    check(r"\xx", EscapeError::InvalidCharInHexEscape);
    check(r"\xы", EscapeError::InvalidCharInHexEscape);
    check(r"\x🦀", EscapeError::InvalidCharInHexEscape);
    check(r"\xtt", EscapeError::InvalidCharInHexEscape);

    check(r"\u", EscapeError::NoBraceInUnicodeEscape);
    check(r"\u[0123]", EscapeError::NoBraceInUnicodeEscape);
    check(r"\u{0x}", EscapeError::InvalidCharInUnicodeEscape);
    check(r"\u{", EscapeError::UnclosedUnicodeEscape);
    check(r"\u{0000", EscapeError::UnclosedUnicodeEscape);
    check(r"\u{}", EscapeError::EmptyUnicodeEscape);
    check(r"\u{_0000}", EscapeError::LeadingUnderscoreUnicodeEscape);
    check(r"\u{0000000}", EscapeError::OverlongUnicodeEscape);

    check("ы", EscapeError::NonAsciiCharInByte);
    check("🦀", EscapeError::NonAsciiCharInByte);

    check(r"\u{0}", EscapeError::UnicodeEscapeInByte);
    check(r"\u{000000}", EscapeError::UnicodeEscapeInByte);
    check(r"\u{41}", EscapeError::UnicodeEscapeInByte);
    check(r"\u{0041}", EscapeError::UnicodeEscapeInByte);
    check(r"\u{00_41}", EscapeError::UnicodeEscapeInByte);
    check(r"\u{4__1__}", EscapeError::UnicodeEscapeInByte);
    check(r"\u{1F63b}", EscapeError::UnicodeEscapeInByte);
    check(r"\u{0}x", EscapeError::UnicodeEscapeInByte);
    check(r"\u{1F63b}}", EscapeError::UnicodeEscapeInByte);
    check(r"\u{FFFFFF}", EscapeError::UnicodeEscapeInByte);
    check(r"\u{ffffff}", EscapeError::UnicodeEscapeInByte);
    check(r"\u{ffffff}", EscapeError::UnicodeEscapeInByte);
    check(r"\u{DC00}", EscapeError::UnicodeEscapeInByte);
    check(r"\u{DDDD}", EscapeError::UnicodeEscapeInByte);
    check(r"\u{DFFF}", EscapeError::UnicodeEscapeInByte);
    check(r"\u{D800}", EscapeError::UnicodeEscapeInByte);
    check(r"\u{DAAA}", EscapeError::UnicodeEscapeInByte);
    check(r"\u{DBFF}", EscapeError::UnicodeEscapeInByte);
}

#[test]
fn test_unescape_byte_good() {
    fn check(literal_text: &str, expected_byte: u8) {
        assert_eq!(unescape_byte(literal_text), Ok(expected_byte));
    }

    check("a", b'a');

    check(r#"\""#, b'"');
    check(r"\n", b'\n');
    check(r"\r", b'\r');
    check(r"\t", b'\t');
    check(r"\\", b'\\');
    check(r"\'", b'\'');
    check(r"\0", b'\0');

    check(r"\x00", b'\0');
    check(r"\x5a", b'Z');
    check(r"\x5A", b'Z');
    check(r"\x7f", 127);
    check(r"\x80", 128);
    check(r"\xff", 255);
    check(r"\xFF", 255);
}

#[test]
fn test_unescape_byte_str_good() {
    fn check(literal_text: &str, expected: &[u8]) {
        let mut buf = Ok(Vec::with_capacity(literal_text.len()));
        unescape_literal(literal_text, Mode::ByteStr, &mut |range, c| {
            if let Ok(b) = &mut buf {
                match c {
                    Ok(c) => b.push(byte_from_char(c)),
                    Err(e) => buf = Err((range, e)),
                }
            }
        });
        assert_eq!(buf.as_deref(), Ok(expected))
    }

    check("foo", b"foo");
    check("", b"");
    check(" \t\n", b" \t\n");

    check("hello \\\n     world", b"hello world");
    check("thread's", b"thread's")
}

#[test]
fn test_unescape_raw_str() {
    fn check(literal: &str, expected: &[(Range<usize>, Result<char, EscapeError>)]) {
        let mut unescaped = Vec::with_capacity(literal.len());
        unescape_literal(literal, Mode::RawStr, &mut |range, res| unescaped.push((range, res)));
        assert_eq!(unescaped, expected);
    }

    check("\r", &[(0..1, Err(EscapeError::BareCarriageReturnInRawString))]);
    check("\rx", &[(0..1, Err(EscapeError::BareCarriageReturnInRawString)), (1..2, Ok('x'))]);
}

#[test]
fn test_unescape_raw_byte_str() {
    fn check(literal: &str, expected: &[(Range<usize>, Result<char, EscapeError>)]) {
        let mut unescaped = Vec::with_capacity(literal.len());
        unescape_literal(literal, Mode::RawByteStr, &mut |range, res| unescaped.push((range, res)));
        assert_eq!(unescaped, expected);
    }

    check("\r", &[(0..1, Err(EscapeError::BareCarriageReturnInRawString))]);
    check("🦀", &[(0..4, Err(EscapeError::NonAsciiCharInByte))]);
    check("🦀a", &[(0..4, Err(EscapeError::NonAsciiCharInByte)), (4..5, Ok('a'))]);
}
