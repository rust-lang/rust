uint_module!(u8);

#[cfg(test)]
mod u8_specific_tests {
    macro_rules! is_ascii_tests {
        ($x: ident $(, $test_name: ident, $method_name: ident, $original_version: expr)+ $(,)?) => {
            $(
                #[test]
                fn $test_name() {
                    for $x in u8::MIN..=u8::MAX {
                        // Test current version against simple Rust 1.63.0 version
                        assert!(
                            $x.$method_name() == ($original_version),
                            concat!("`{}_u8.", stringify!($variant), "()` is incorrect"),
                            $x
                        );
                    }
                }
            )*
        };
    }

    is_ascii_tests!(
        x,
        test_is_ascii,
        is_ascii,
        x & 128 == 0,
        test_is_ascii_alphabetic,
        is_ascii_alphabetic,
        matches!(x, b'A'..=b'Z' | b'a'..=b'z'),
        test_is_ascii_alphanumeric,
        is_ascii_alphanumeric,
        matches!(x, b'0'..=b'9' | b'A'..=b'Z' | b'a'..=b'z'),
        test_is_ascii_control,
        is_ascii_control,
        matches!(x, b'\0'..=b'\x1F' | b'\x7F'),
        test_is_ascii_digit,
        is_ascii_digit,
        matches!(x, b'0'..=b'9'),
        test_is_ascii_graphic,
        is_ascii_graphic,
        matches!(x, b'!'..=b'~'),
        test_is_ascii_hexdigit,
        is_ascii_hexdigit,
        matches!(x, b'0'..=b'9' | b'A'..=b'F' | b'a'..=b'f'),
        test_is_ascii_lowercase,
        is_ascii_lowercase,
        matches!(x, b'a'..=b'z'),
        test_is_ascii_punctuation,
        is_ascii_punctuation,
        matches!(x, b'!'..=b'/' | b':'..=b'@' | b'['..=b'`' | b'{'..=b'~'),
        test_is_ascii_uppercase,
        is_ascii_uppercase,
        matches!(x, b'A'..=b'Z'),
        test_is_ascii_whitespace,
        is_ascii_whitespace,
        matches!(x, b'\t' | b'\n' | b'\x0C' | b'\r' | b' '),
    );
}
