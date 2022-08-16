uint_module!(u8);

#[cfg(test)]
mod u8_specific_tests {
    #[test]
    fn is_ascii_tests() {
        for ch in u8::MIN..=u8::MAX {
            let ascii = matches!(ch, 0..=0x7f);
            let control = matches!(ch, 0..=31 | 0x7f);
            let digit = matches!(ch, b'0'..=b'9');
            let graphic = matches!(ch, b'!'..=b'~');
            let hex_letter = matches!(ch, b'A'..=b'F' | b'a'..=b'f');
            let lowercase = matches!(ch, b'a'..=b'z');
            let punctuation = matches!(ch, b'!'..=b'/' | b':'..=b'@' | b'['..=b'`' | b'{'..=b'~');
            let uppercase = matches!(ch, b'A'..=b'Z');
            let whitespace = matches!(ch, b'\t' | b'\n' | b'\r' | b'\x0c' | b' ');

            let escaped_ch = ch.escape_ascii();

            assert_eq!(ch.is_ascii(), ascii, "b'{escaped_ch}'.is_ascii() is incorrect");
            assert_eq!(
                ch.is_ascii_alphabetic(),
                uppercase | lowercase,
                "b'{escaped_ch}'.is_ascii_alphabetic() is incorrect"
            );
            assert_eq!(
                ch.is_ascii_alphanumeric(),
                uppercase | lowercase | digit,
                "b'{escaped_ch}'.is_ascii_alphanumeric() is incorrect"
            );
            assert_eq!(
                ch.is_ascii_control(),
                control,
                "b'{escaped_ch}'.is_ascii_control() is incorrect"
            );
            assert_eq!(ch.is_ascii_digit(), digit, "b'{escaped_ch}'.is_ascii_digit() is incorrect");
            assert_eq!(
                ch.is_ascii_graphic(),
                graphic,
                "b'{escaped_ch}'.is_ascii_graphic() is incorrect"
            );
            assert_eq!(
                ch.is_ascii_hexdigit(),
                digit | hex_letter,
                "b'{escaped_ch}'.is_ascii_hexdigit() is incorrect"
            );
            assert_eq!(
                ch.is_ascii_lowercase(),
                lowercase,
                "b'{escaped_ch}'.is_ascii_lowercase() is incorrect"
            );
            assert_eq!(
                ch.is_ascii_punctuation(),
                punctuation,
                "b'{escaped_ch}'.is_ascii_punctuation() is incorrect"
            );
            assert_eq!(
                ch.is_ascii_uppercase(),
                uppercase,
                "b'{escaped_ch}'.is_ascii_uppercase() is incorrect"
            );
            assert_eq!(
                ch.is_ascii_whitespace(),
                whitespace,
                "b'{escaped_ch}'.is_ascii_whitespace() is incorrect"
            );
        }
    }
}
