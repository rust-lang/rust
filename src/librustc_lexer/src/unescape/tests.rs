use super::*;

#[test]
fn test_unescape_char_bad() {
    fn check(literal_text: &str, expected_error: EscapeError) {
        let actual_result = unescape_char(literal_text).map_err(|(_offset, err)| err);
        assert_eq!(actual_result, Err(expected_error));
    }

    check("", EscapeError::ZeroChars);
    check(r"\", EscapeError::LoneSlash);

    check("\n", EscapeError::EscapeOnlyChar);
    check("\r\n", EscapeError::EscapeOnlyChar);
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
        let actual_result = unescape_char(literal_text);
        assert_eq!(actual_result, Ok(expected_char));
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
fn test_unescape_str_good() {
    fn check(literal_text: &str, expected: &str) {
        let mut buf = Ok(String::with_capacity(literal_text.len()));
        unescape_str(literal_text, &mut |range, c| {
            if let Ok(b) = &mut buf {
                match c {
                    Ok(c) => b.push(c),
                    Err(e) => buf = Err((range, e)),
                }
            }
        });
        let buf = buf.as_ref().map(|it| it.as_ref());
        assert_eq!(buf, Ok(expected))
    }

    check("foo", "foo");
    check("", "");
    check(" \t\n\r\n", " \t\n\n");

    check("hello \\\n     world", "hello world");
    check("hello \\\r\n     world", "hello world");
    check("thread's", "thread's")
}

#[test]
fn test_unescape_byte_bad() {
    fn check(literal_text: &str, expected_error: EscapeError) {
        let actual_result = unescape_byte(literal_text).map_err(|(_offset, err)| err);
        assert_eq!(actual_result, Err(expected_error));
    }

    check("", EscapeError::ZeroChars);
    check(r"\", EscapeError::LoneSlash);

    check("\n", EscapeError::EscapeOnlyChar);
    check("\r\n", EscapeError::EscapeOnlyChar);
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
        let actual_result = unescape_byte(literal_text);
        assert_eq!(actual_result, Ok(expected_byte));
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
        unescape_byte_str(literal_text, &mut |range, c| {
            if let Ok(b) = &mut buf {
                match c {
                    Ok(c) => b.push(c),
                    Err(e) => buf = Err((range, e)),
                }
            }
        });
        let buf = buf.as_ref().map(|it| it.as_ref());
        assert_eq!(buf, Ok(expected))
    }

    check("foo", b"foo");
    check("", b"");
    check(" \t\n\r\n", b" \t\n\n");

    check("hello \\\n     world", b"hello world");
    check("hello \\\r\n     world", b"hello world");
    check("thread's", b"thread's")
}

#[test]
fn test_unescape_raw_str() {
    fn check(literal: &str, expected: &[(Range<usize>, Result<char, EscapeError>)]) {
        let mut unescaped = Vec::with_capacity(literal.len());
        unescape_raw_str(literal, &mut |range, res| unescaped.push((range, res)));
        assert_eq!(unescaped, expected);
    }

    check("\r\n", &[(0..2, Ok('\n'))]);
    check("\r", &[(0..1, Err(EscapeError::BareCarriageReturnInRawString))]);
    check("\rx", &[(0..1, Err(EscapeError::BareCarriageReturnInRawString)), (1..2, Ok('x'))]);
}

#[test]
fn test_unescape_raw_byte_str() {
    fn check(literal: &str, expected: &[(Range<usize>, Result<u8, EscapeError>)]) {
        let mut unescaped = Vec::with_capacity(literal.len());
        unescape_raw_byte_str(literal, &mut |range, res| unescaped.push((range, res)));
        assert_eq!(unescaped, expected);
    }

    check("\r\n", &[(0..2, Ok(byte_from_char('\n')))]);
    check("\r", &[(0..1, Err(EscapeError::BareCarriageReturnInRawString))]);
    check("🦀", &[(0..4, Err(EscapeError::NonAsciiCharInByteString))]);
    check(
        "🦀a",
        &[(0..4, Err(EscapeError::NonAsciiCharInByteString)), (4..5, Ok(byte_from_char('a')))],
    );
}
