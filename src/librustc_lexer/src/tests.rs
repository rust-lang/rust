use super::*;

fn check_raw_str(s: &str, expected_hashes: u16, expected_err: Option<RawStrError>) {
    let s = &format!("r{}", s);
    let mut cursor = Cursor::new(s);
    cursor.bump();
    let (n_hashes, err) = cursor.raw_double_quoted_string(0);
    assert_eq!(n_hashes, expected_hashes);
    assert_eq!(err, expected_err);
}

#[test]
fn test_naked_raw_str() {
    check_raw_str(r#""abc""#, 0, None);
}

#[test]
fn test_raw_no_start() {
    check_raw_str(r##""abc"#"##, 0, None);
}

#[test]
fn test_too_many_terminators() {
    // this error is handled in the parser later
    check_raw_str(r###"#"abc"##"###, 1, None);
}

#[test]
fn test_unterminated() {
    check_raw_str(
        r#"#"abc"#,
        1,
        Some(RawStrError::NoTerminator { expected: 1, found: 0, possible_terminator_offset: None }),
    );
    check_raw_str(
        r###"##"abc"#"###,
        2,
        Some(RawStrError::NoTerminator {
            expected: 2,
            found: 1,
            possible_terminator_offset: Some(7),
        }),
    );
    // We're looking for "# not just any #
    check_raw_str(
        r###"##"abc#"###,
        2,
        Some(RawStrError::NoTerminator { expected: 2, found: 0, possible_terminator_offset: None }),
    )
}

#[test]
fn test_invalid_start() {
    check_raw_str(r##"#~"abc"#"##, 1, Some(RawStrError::InvalidStarter { bad_char: '~' }));
}

#[test]
fn test_unterminated_no_pound() {
    // https://github.com/rust-lang/rust/issues/70677
    check_raw_str(
        r#"""#,
        0,
        Some(RawStrError::NoTerminator { expected: 0, found: 0, possible_terminator_offset: None }),
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
