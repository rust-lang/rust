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
                possible_terminator_offset: None,
            },
            Err(LexRawStrError::InvalidStarter),
        );
    }
}
