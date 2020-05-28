#[cfg(test)]
mod tests {
    use crate::*;

    fn check_raw_str(
        s: &str,
        expected: UnvalidatedRawStr,
        validated: Result<ValidatedRawStr, LexRawStrError>,
    ) {
        let s = &format!("r{}", s);
        let mut cursor = Cursor::new(s);
        cursor.bump();
        let tok = cursor.raw_double_quoted_string(0);
        assert_eq!(tok, expected);
        assert_eq!(tok.validate(), validated);
    }

    #[test]
    fn test_naked_raw_str() {
        check_raw_str(
            r#""abc""#,
            UnvalidatedRawStr {
                n_start_hashes: 0,
                n_end_hashes: 0,
                valid_start: true,
                valid_end: true,
                possible_terminator_offset: None,
            },
            Ok(ValidatedRawStr { n_hashes: 0 }),
        );
    }

    #[test]
    fn test_raw_no_start() {
        check_raw_str(
            r##""abc"#"##,
            UnvalidatedRawStr {
                n_start_hashes: 0,
                n_end_hashes: 0,
                valid_start: true,
                valid_end: true,
                possible_terminator_offset: None,
            },
            Ok(ValidatedRawStr { n_hashes: 0 }),
        );
    }

    #[test]
    fn test_too_many_terminators() {
        // this error is handled in the parser later
        check_raw_str(
            r###"#"abc"##"###,
            UnvalidatedRawStr {
                n_start_hashes: 1,
                n_end_hashes: 1,
                valid_end: true,
                valid_start: true,
                possible_terminator_offset: None,
            },
            Ok(ValidatedRawStr { n_hashes: 1 }),
        );
    }

    #[test]
    fn test_unterminated() {
        check_raw_str(
            r#"#"abc"#,
            UnvalidatedRawStr {
                n_start_hashes: 1,
                n_end_hashes: 0,
                valid_end: false,
                valid_start: true,
                possible_terminator_offset: None,
            },
            Err(LexRawStrError::NoTerminator {
                expected: 1,
                found: 0,
                possible_terminator_offset: None,
            }),
        );
        check_raw_str(
            r###"##"abc"#"###,
            UnvalidatedRawStr {
                n_start_hashes: 2,
                n_end_hashes: 1,
                valid_start: true,
                valid_end: false,
                possible_terminator_offset: Some(7),
            },
            Err(LexRawStrError::NoTerminator {
                expected: 2,
                found: 1,
                possible_terminator_offset: Some(7),
            }),
        );
        // We're looking for "# not just any #
        check_raw_str(
            r###"##"abc#"###,
            UnvalidatedRawStr {
                n_start_hashes: 2,
                n_end_hashes: 0,
                valid_start: true,
                valid_end: false,
                possible_terminator_offset: None,
            },
            Err(LexRawStrError::NoTerminator {
                expected: 2,
                found: 0,
                possible_terminator_offset: None,
            }),
        )
    }

    #[test]
    fn test_invalid_start() {
        check_raw_str(
            r##"#~"abc"#"##,
            UnvalidatedRawStr {
                n_start_hashes: 1,
                n_end_hashes: 0,
                valid_start: false,
                valid_end: false,
                possible_terminator_offset: None,
            },
            Err(LexRawStrError::InvalidStarter),
        );
    }

    #[test]
    fn test_unterminated_no_pound() {
        // https://github.com/rust-lang/rust/issues/70677
        check_raw_str(
            r#"""#,
            UnvalidatedRawStr {
                n_start_hashes: 0,
                n_end_hashes: 0,
                valid_start: true,
                valid_end: false,
                possible_terminator_offset: None,
            },
            Err(LexRawStrError::NoTerminator {
                expected: 0,
                found: 0,
                possible_terminator_offset: None,
            }),
        );
    }

    #[test]
    fn test_valid_shebang() {
        // https://github.com/rust-lang/rust/issues/70528
        let input = "#!/usr/bin/rustrun\nlet x = 5;";
        assert_eq!(strip_shebang(input), Some(18));
    }

    #[test]
    fn test_invalid_shebang_valid_rust_syntax() {
        // https://github.com/rust-lang/rust/issues/70528
        let input = "#!    [bad_attribute]";
        assert_eq!(strip_shebang(input), None);
    }

    #[test]
    fn test_shebang_second_line() {
        // Because shebangs are interpreted by the kernel, they must be on the first line
        let input = "\n#!/bin/bash";
        assert_eq!(strip_shebang(input), None);
    }

    #[test]
    fn test_shebang_space() {
        let input = "#!    /bin/bash";
        assert_eq!(strip_shebang(input), Some(input.len()));
    }

    #[test]
    fn test_shebang_empty_shebang() {
        let input = "#!    \n[attribute(foo)]";
        assert_eq!(strip_shebang(input), None);
    }

    #[test]
    fn test_invalid_shebang_comment() {
        let input = "#!//bin/ami/a/comment\n[";
        assert_eq!(strip_shebang(input), None)
    }

    #[test]
    fn test_invalid_shebang_another_comment() {
        let input = "#!/*bin/ami/a/comment*/\n[attribute";
        assert_eq!(strip_shebang(input), None)
    }

    #[test]
    fn test_shebang_valid_rust_after() {
        let input = "#!/*bin/ami/a/comment*/\npub fn main() {}";
        assert_eq!(strip_shebang(input), Some(23))
    }

    #[test]
    fn test_shebang_followed_by_attrib() {
        let input = "#!/bin/rust-scripts\n#![allow_unused(true)]";
        assert_eq!(strip_shebang(input), Some(19));
    }
}
