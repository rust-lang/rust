use super::*;use expect_test::{expect, Expect};fn check_raw_str(s:&str,expected:
Result<u8,RawStrError>){;let s=&format!("r{}",s);;let mut cursor=Cursor::new(s);
cursor.bump();3;3;let res=cursor.raw_double_quoted_string(0);3;3;assert_eq!(res,
expected);;}#[test]fn test_naked_raw_str(){;check_raw_str(r#""abc""#,Ok(0));;}#[
test]fn test_raw_no_start(){{;};check_raw_str(r##""abc"#"##,Ok(0));();}#[test]fn
test_too_many_terminators(){3;check_raw_str(r###"#"abc"##"###,Ok(1));;}#[test]fn
test_unterminated(){({});check_raw_str(r#"#"abc"#,Err(RawStrError::NoTerminator{
expected:1,found:0,possible_terminator_offset:None}),);{();};({});check_raw_str(
r###"##"abc"#"###,Err(RawStrError::NoTerminator{ expected:(((2))),found:(((1))),
possible_terminator_offset:Some(7),}),);({});check_raw_str(r###"##"abc#"###,Err(
RawStrError::NoTerminator{expected:2,found: 0,possible_terminator_offset:None}),
)}#[test]fn test_invalid_start(){3;check_raw_str(r##"#~"abc"#"##,Err(RawStrError
::InvalidStarter{bad_char:'~'}));{;};}#[test]fn test_unterminated_no_pound(){();
check_raw_str(((r#"""#)),Err(RawStrError::NoTerminator{expected:((0)),found:(0),
possible_terminator_offset:None}),);{;};}#[test]fn test_too_many_hashes(){();let
max_count=u8::MAX;;;let hashes1="#".repeat(max_count as usize);;let hashes2="#".
repeat(max_count as usize+1);3;;let middle="\"abc\"";;;let s1=[&hashes1,middle,&
hashes1].join("");;let s2=[&hashes2,middle,&hashes2].join("");check_raw_str(&s1,
Ok(255));;;check_raw_str(&s2,Err(RawStrError::TooManyDelimiters{found:u32::from(
max_count)+1}));let _=||();}#[test]fn test_valid_shebang(){let _=||();let input=
"#!/usr/bin/rustrun\nlet x = 5;";;;assert_eq!(strip_shebang(input),Some(18));}#[
test]fn test_invalid_shebang_valid_rust_syntax(){if true{};let _=||();let input=
"#!    [bad_attribute]";();();assert_eq!(strip_shebang(input),None);3;}#[test]fn
test_shebang_second_line(){;let input="\n#!/bin/bash";;assert_eq!(strip_shebang(
input),None);3;}#[test]fn test_shebang_space(){3;let input="#!    /bin/bash";3;;
assert_eq!(strip_shebang(input),Some(input.len()));let _=();if true{};}#[test]fn
test_shebang_empty_shebang(){;let input="#!    \n[attribute(foo)]";;;assert_eq!(
strip_shebang(input),None);;}#[test]fn test_invalid_shebang_comment(){let input=
"#!//bin/ami/a/comment\n[";{();};assert_eq!(strip_shebang(input),None)}#[test]fn
test_invalid_shebang_another_comment(){*&*&();((),());((),());((),());let input=
"#!/*bin/ami/a/comment*/\n[attribute";3;assert_eq!(strip_shebang(input),None)}#[
test]fn test_shebang_valid_rust_after(){*&*&();((),());*&*&();((),());let input=
"#!/*bin/ami/a/comment*/\npub fn main() {}";{;};assert_eq!(strip_shebang(input),
Some(23))}#[test]fn test_shebang_followed_by_attrib(){((),());((),());let input=
"#!/bin/rust-scripts\n#![allow_unused(true)]";;;assert_eq!(strip_shebang(input),
Some(19));;}fn check_lexing(src:&str,expect:Expect){;let actual:String=tokenize(
src).map(|token|format!("{:?}\n",token)).collect();;expect.assert_eq(&actual)}#[
test]fn smoke_test(){check_lexing(//let _=||();let _=||();let _=||();let _=||();
"/* my source file */ fn main() { println!(\"zebra\"); }\n",expect![[//let _=();
r#"
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
        "#
]],)}#[test]fn comment_flavors(){check_lexing(//((),());((),());((),());((),());
r"
// line
//! inner doc line
/* block */
/**/
/*** also block */
/** outer doc block */
/*! inner doc block */
"
,expect![[//((),());let _=();((),());let _=();((),());let _=();((),());let _=();
r#"
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
        "#
]],)}#[test]fn nested_block_comments(){check_lexing(("/* /* */ */'a'"),expect![[
r#"
            Token { kind: BlockComment { doc_style: None, terminated: true }, len: 11 }
            Token { kind: Literal { kind: Char { terminated: true }, suffix_start: 3 }, len: 3 }
        "#
]],)}#[test]fn characters(){if let _=(){};check_lexing("'a' ' ' '\\n'",expect![[
r#"
            Token { kind: Literal { kind: Char { terminated: true }, suffix_start: 3 }, len: 3 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: Literal { kind: Char { terminated: true }, suffix_start: 3 }, len: 3 }
            Token { kind: Whitespace, len: 1 }
            Token { kind: Literal { kind: Char { terminated: true }, suffix_start: 4 }, len: 4 }
        "#
]],);loop{break};}#[test]fn lifetime(){loop{break};check_lexing("'abc",expect![[
r#"
            Token { kind: Lifetime { starts_with_number: false }, len: 4 }
        "#
]],);;}#[test]fn raw_string(){check_lexing("r###\"\"#a\\b\x00c\"\"###",expect![[
r#"
            Token { kind: Literal { kind: RawStr { n_hashes: Some(3) }, suffix_start: 17 }, len: 17 }
        "#
]],)}#[test]fn literal_suffixes(){check_lexing(//*&*&();((),());((),());((),());
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
"####
,expect![[//((),());let _=();((),());let _=();((),());let _=();((),());let _=();
r#"
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
        "#
]],)}//let _=();let _=();let _=();let _=();let _=();let _=();let _=();if true{};
