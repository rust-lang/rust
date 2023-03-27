use super::*;

use expect_test::{expect, Expect};

fn check_raw_str(s: &str, expected: Result<u8, RawStrError>) {
    let s = &format!("r{}", s);
    let mut lexer = Lexer::new(s);
    lexer.cursor.bump();
    let res = lexer.raw_double_quoted_string(0);
    assert_eq!(res, expected);
}

#[test]
fn test_naked_raw_str() {
    check_raw_str(r#""abc""#, Ok(0));
}

#[test]
fn test_raw_no_start() {
    check_raw_str(r##""abc"#"##, Ok(0));
}

#[test]
fn test_too_many_terminators() {
    // this error is handled in the parser later
    check_raw_str(r###"#"abc"##"###, Ok(1));
}

#[test]
fn test_unterminated() {
    check_raw_str(
        r#"#"abc"#,
        Err(RawStrError::NoTerminator { expected: 1, found: 0, possible_terminator_offset: None }),
    );
    check_raw_str(
        r###"##"abc"#"###,
        Err(RawStrError::NoTerminator {
            expected: 2,
            found: 1,
            possible_terminator_offset: Some(7),
        }),
    );
    // We're looking for "# not just any #
    check_raw_str(
        r###"##"abc#"###,
        Err(RawStrError::NoTerminator { expected: 2, found: 0, possible_terminator_offset: None }),
    )
}

#[test]
fn test_invalid_start() {
    check_raw_str(r##"#~"abc"#"##, Err(RawStrError::InvalidStarter { bad_char: '~' }));
}

#[test]
fn test_unterminated_no_pound() {
    // https://github.com/rust-lang/rust/issues/70677
    check_raw_str(
        r#"""#,
        Err(RawStrError::NoTerminator { expected: 0, found: 0, possible_terminator_offset: None }),
    );
}

#[test]
fn test_too_many_hashes() {
    let max_count = u8::MAX;
    let hashes1 = "#".repeat(max_count as usize);
    let hashes2 = "#".repeat(max_count as usize + 1);
    let middle = "\"abc\"";
    let s1 = [&hashes1, middle, &hashes1].join("");
    let s2 = [&hashes2, middle, &hashes2].join("");

    // Valid number of hashes (255 = 2^8 - 1 = u8::MAX).
    check_raw_str(&s1, Ok(255));

    // One more hash sign (256 = 2^8) becomes too many.
    check_raw_str(&s2, Err(RawStrError::TooManyDelimiters { found: u32::from(max_count) + 1 }));
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

fn check_lexing(src: &str, expect: Expect) {
    let actual: String = tokenize(src).map(|token| format!("{:?}\n", token)).collect();
    expect.assert_eq(&actual)
}

#[test]
fn smoke_test() {
    check_lexing(
        "/* my source file */ fn main() { println!(\"zebra\"); }\n",
        expect![[r#"
            Token { kind: BlockComment { doc_style: None, terminated: true }, len: 20 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: Ident, len: 2 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: Ident, len: 4 }
            Token { kind: OpenParen, len: 1 }
            Token { kind: CloseParen, len: 1 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: OpenBrace, len: 1 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: Ident, len: 7 }
            Token { kind: Bang, len: 1 }
            Token { kind: OpenParen, len: 1 }
            Token { kind: Literal { kind: Str { terminated: true }, suffix_start: 7 }, len: 7 }
            Token { kind: CloseParen, len: 1 }
            Token { kind: Semi, len: 1 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: CloseBrace, len: 1 }
            Token { kind: Whitespace, len: 1 }
        "#]],
    )
}

#[test]
fn comment_flavors() {
    check_lexing(
        r"
// line
//// line as well
/// outer doc line
//! inner doc line
/* block */
/**/
/*** also block */
/** outer doc block */
/*! inner doc block */
",
        expect![[r#"
            Token { kind: Whitespace, len: 1 }
            Token { kind: LineComment { doc_style: None }, len: 7 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: LineComment { doc_style: None }, len: 17 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: LineComment { doc_style: Some(Outer) }, len: 18 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: LineComment { doc_style: Some(Inner) }, len: 18 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: BlockComment { doc_style: None, terminated: true }, len: 11 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: BlockComment { doc_style: None, terminated: true }, len: 4 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: BlockComment { doc_style: None, terminated: true }, len: 18 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: BlockComment { doc_style: Some(Outer), terminated: true }, len: 22 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: BlockComment { doc_style: Some(Inner), terminated: true }, len: 22 }
            Token { kind: Whitespace, len: 1 }
        "#]],
    )
}

#[test]
fn nested_block_comments() {
    check_lexing(
        "/* /* */ */'a'",
        expect![[r#"
            Token { kind: BlockComment { doc_style: None, terminated: true }, len: 11 }
            Token { kind: Literal { kind: Char { terminated: true }, suffix_start: 3 }, len: 3 }
        "#]],
    )
}

#[test]
fn characters() {
    check_lexing(
        "'a' ' ' '\\n'",
        expect![[r#"
            Token { kind: Literal { kind: Char { terminated: true }, suffix_start: 3 }, len: 3 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: Literal { kind: Char { terminated: true }, suffix_start: 3 }, len: 3 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: Literal { kind: Char { terminated: true }, suffix_start: 4 }, len: 4 }
        "#]],
    );
}

#[test]
fn lifetime() {
    check_lexing(
        "'abc",
        expect![[r#"
            Token { kind: Lifetime { starts_with_number: false, contains_emoji: false }, len: 4 }
        "#]],
    );
}

#[test]
fn raw_string() {
    check_lexing(
        "r###\"\"#a\\b\x00c\"\"###",
        expect![[r#"
            Token { kind: Literal { kind: RawStr { n_hashes: Some(3) }, suffix_start: 17 }, len: 17 }
        "#]],
    )
}

#[test]
fn literal_suffixes() {
    check_lexing(
        r####"
'a'
b'a'
"a"
b"a"
1234
0b101
0xABC
1.0
1.0e10
2us
r###"raw"###suffix
br###"raw"###suffix
f"fstr"not_suffix
f"foo{not_a_suffix + 2}bar"not_suffix
"####,
        expect![[r#"
            Token { kind: Whitespace, len: 1 }
            Token { kind: Literal { kind: Char { terminated: true }, suffix_start: 3 }, len: 3 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: Literal { kind: Byte { terminated: true }, suffix_start: 4 }, len: 4 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: Literal { kind: Str { terminated: true }, suffix_start: 3 }, len: 3 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: Literal { kind: ByteStr { terminated: true }, suffix_start: 4 }, len: 4 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: Literal { kind: Int { base: Decimal, empty_int: false }, suffix_start: 4 }, len: 4 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: Literal { kind: Int { base: Binary, empty_int: false }, suffix_start: 5 }, len: 5 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: Literal { kind: Int { base: Hexadecimal, empty_int: false }, suffix_start: 5 }, len: 5 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: Literal { kind: Float { base: Decimal, empty_exponent: false }, suffix_start: 3 }, len: 3 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: Literal { kind: Float { base: Decimal, empty_exponent: false }, suffix_start: 6 }, len: 6 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: Literal { kind: Int { base: Decimal, empty_int: false }, suffix_start: 1 }, len: 3 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: Literal { kind: RawStr { n_hashes: Some(3) }, suffix_start: 12 }, len: 18 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: Literal { kind: RawByteStr { n_hashes: Some(3) }, suffix_start: 13 }, len: 19 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: Literal { kind: FStr { start: Quote, end: Some(Quote) }, suffix_start: 7 }, len: 13 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: Literal { kind: FStr { start: Quote, end: Some(Brace) }, suffix_start: 6 }, len: 6 }
            Token { kind: Ident, len: 12 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: Plus, len: 1 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: Literal { kind: Int { base: Decimal, empty_int: false }, suffix_start: 1 }, len: 1 }
            Token { kind: Literal { kind: FStr { start: Brace, end: Some(Quote) }, suffix_start: 5 }, len: 11 }
            Token { kind: Whitespace, len: 1 }
        "#]],
    )
}

#[test]
fn f_string() {
    check_lexing(
        r#"
f"foobar"
f"foo{ 1 + 1 }bar"
f"foo{ 1 + { 5 } }bar"
f"foo{ 1 + { 5 + { ident } } }bar"
"#,
        expect![[r#"
            Token { kind: Whitespace, len: 1 }
            Token { kind: Literal { kind: FStr { start: Quote, end: Some(Quote) }, suffix_start: 9 }, len: 9 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: Literal { kind: FStr { start: Quote, end: Some(Brace) }, suffix_start: 6 }, len: 6 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: Literal { kind: Int { base: Decimal, empty_int: false }, suffix_start: 1 }, len: 1 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: Plus, len: 1 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: Literal { kind: Int { base: Decimal, empty_int: false }, suffix_start: 1 }, len: 1 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: Literal { kind: FStr { start: Brace, end: Some(Quote) }, suffix_start: 5 }, len: 5 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: Literal { kind: FStr { start: Quote, end: Some(Brace) }, suffix_start: 6 }, len: 6 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: Literal { kind: Int { base: Decimal, empty_int: false }, suffix_start: 1 }, len: 1 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: Plus, len: 1 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: OpenBrace, len: 1 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: Literal { kind: Int { base: Decimal, empty_int: false }, suffix_start: 1 }, len: 1 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: CloseBrace, len: 1 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: Literal { kind: FStr { start: Brace, end: Some(Quote) }, suffix_start: 5 }, len: 5 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: Literal { kind: FStr { start: Quote, end: Some(Brace) }, suffix_start: 6 }, len: 6 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: Literal { kind: Int { base: Decimal, empty_int: false }, suffix_start: 1 }, len: 1 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: Plus, len: 1 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: OpenBrace, len: 1 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: Literal { kind: Int { base: Decimal, empty_int: false }, suffix_start: 1 }, len: 1 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: Plus, len: 1 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: OpenBrace, len: 1 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: Ident, len: 5 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: CloseBrace, len: 1 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: CloseBrace, len: 1 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: Literal { kind: FStr { start: Brace, end: Some(Quote) }, suffix_start: 5 }, len: 5 }
            Token { kind: Whitespace, len: 1 }
        "#]],
    )
}

#[test]
fn f_string_escaped() {
    check_lexing(
        r#"
f"foobar"
f"foo\{ quux \}bar"
f"foo{{ quux {{\{}}\}\\}}bar"
f"foo\{"
f"foo\}"
f"foo\\"
f"foo\'"
f"foo\""
f"this whole{{string should}}be one\\}}\{\}literal\\{{"
f"foo{{ bar }} bax: { 1 + 1 }\}\}"
f"foo{{ bar }} bax: { f"inner\{\}{{" }\}\}"
"#,
        expect![[r#"
            Token { kind: Whitespace, len: 1 }
            Token { kind: Literal { kind: FStr { start: Quote, end: Some(Quote) }, suffix_start: 9 }, len: 9 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: Literal { kind: FStr { start: Quote, end: Some(Quote) }, suffix_start: 19 }, len: 19 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: Literal { kind: FStr { start: Quote, end: Some(Quote) }, suffix_start: 29 }, len: 29 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: Literal { kind: FStr { start: Quote, end: Some(Quote) }, suffix_start: 8 }, len: 8 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: Literal { kind: FStr { start: Quote, end: Some(Quote) }, suffix_start: 8 }, len: 8 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: Literal { kind: FStr { start: Quote, end: Some(Quote) }, suffix_start: 8 }, len: 8 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: Literal { kind: FStr { start: Quote, end: Some(Quote) }, suffix_start: 8 }, len: 8 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: Literal { kind: FStr { start: Quote, end: Some(Quote) }, suffix_start: 8 }, len: 8 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: Literal { kind: FStr { start: Quote, end: Some(Quote) }, suffix_start: 55 }, len: 55 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: Literal { kind: FStr { start: Quote, end: Some(Brace) }, suffix_start: 21 }, len: 21 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: Literal { kind: Int { base: Decimal, empty_int: false }, suffix_start: 1 }, len: 1 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: Plus, len: 1 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: Literal { kind: Int { base: Decimal, empty_int: false }, suffix_start: 1 }, len: 1 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: Literal { kind: FStr { start: Brace, end: Some(Quote) }, suffix_start: 6 }, len: 6 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: Literal { kind: FStr { start: Quote, end: Some(Brace) }, suffix_start: 21 }, len: 21 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: Literal { kind: FStr { start: Quote, end: Some(Quote) }, suffix_start: 14 }, len: 14 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: Literal { kind: FStr { start: Brace, end: Some(Quote) }, suffix_start: 6 }, len: 6 }
            Token { kind: Whitespace, len: 1 }
        "#]],
    )
}

#[test]
fn f_string_large() {
    check_lexing(
        r#"f"foo{ ident + f"nested { bar + "f-strings" + f"just }} a plain {{ string" }" }bar""#,
        expect![[r#"
            Token { kind: Literal { kind: FStr { start: Quote, end: Some(Brace) }, suffix_start: 6 }, len: 6 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: Ident, len: 5 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: Plus, len: 1 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: Literal { kind: FStr { start: Quote, end: Some(Brace) }, suffix_start: 10 }, len: 10 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: Ident, len: 3 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: Plus, len: 1 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: Literal { kind: Str { terminated: true }, suffix_start: 11 }, len: 11 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: Plus, len: 1 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: Literal { kind: FStr { start: Quote, end: Some(Quote) }, suffix_start: 28 }, len: 28 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: Literal { kind: FStr { start: Brace, end: Some(Quote) }, suffix_start: 2 }, len: 2 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: Literal { kind: FStr { start: Brace, end: Some(Quote) }, suffix_start: 5 }, len: 5 }
        "#]],
    )
}

#[test]
fn f_string_unterminated() {
    check_lexing(
        r#"f"foo\{"#,
        expect![[r#"
            Token { kind: Literal { kind: FStr { start: Quote, end: None }, suffix_start: 7 }, len: 7 }
        "#]],
    )
}

#[test]
fn f_string_unterminated_with_inner_expr() {
    check_lexing(
        r#"f"foo{1 + 2}"#,
        expect![[r#"
            Token { kind: Literal { kind: FStr { start: Quote, end: Some(Brace) }, suffix_start: 6 }, len: 6 }
            Token { kind: Literal { kind: Int { base: Decimal, empty_int: false }, suffix_start: 1 }, len: 1 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: Plus, len: 1 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: Literal { kind: Int { base: Decimal, empty_int: false }, suffix_start: 1 }, len: 1 }
            Token { kind: Literal { kind: FStr { start: Brace, end: None }, suffix_start: 1 }, len: 1 }
        "#]],
    )
}

#[test]
fn f_string_multiline() {
    check_lexing(
        r#"f"foo{
1 + 2
}bar""#,
        expect![[r#"
            Token { kind: Literal { kind: FStr { start: Quote, end: Some(Brace) }, suffix_start: 6 }, len: 6 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: Literal { kind: Int { base: Decimal, empty_int: false }, suffix_start: 1 }, len: 1 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: Plus, len: 1 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: Literal { kind: Int { base: Decimal, empty_int: false }, suffix_start: 1 }, len: 1 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: Literal { kind: FStr { start: Brace, end: Some(Quote) }, suffix_start: 5 }, len: 5 }
        "#]],
    )
}

#[test]
fn f_string_inner_string_literal() {
    check_lexing(
        r#"f"foo { "bar}" } bar""#,
        expect![[r#"
            Token { kind: Literal { kind: FStr { start: Quote, end: Some(Brace) }, suffix_start: 7 }, len: 7 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: Literal { kind: Str { terminated: true }, suffix_start: 6 }, len: 6 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: Literal { kind: FStr { start: Brace, end: Some(Quote) }, suffix_start: 6 }, len: 6 }
        "#]],
    )
}
