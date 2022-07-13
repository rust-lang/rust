use std::path::Path;

use crate::parser::Pattern;

use super::Comments;

#[test]
fn parse_simple_comment() {
    let s = r"
use std::mem;

fn main() {
    let _x: &i32 = unsafe { mem::transmute(16usize) }; //~ ERROR: encountered a dangling reference (address $HEX is unallocated)
}
    ";
    let comments = Comments::parse(Path::new("<dummy>"), s).unwrap();
    println!("parsed comments: {:#?}", comments);
    assert_eq!(comments.error_matches[0].definition_line, 5);
    assert_eq!(comments.error_matches[0].revision, None);
    match &comments.error_matches[0].pattern {
        Pattern::SubString(s) =>
            assert_eq!(s, "encountered a dangling reference (address $HEX is unallocated)"),
        other => panic!("expected substring, got {other:?}"),
    }
}

#[test]
fn parse_missing_level() {
    let s = r"
use std::mem;

fn main() {
    let _x: &i32 = unsafe { mem::transmute(16usize) }; //~ encountered a dangling reference (address $HEX is unallocated)
}
    ";
    assert!(Comments::parse(Path::new("<dummy>"), s).is_err(), "expected parsing to fail");
}

#[test]
fn parse_slash_slash_at() {
    let s = r"
//@  error-pattern:  foomp
use std::mem;

    ";
    let comments = Comments::parse(Path::new("<dummy>"), s).unwrap();
    println!("parsed comments: {:#?}", comments);
    let pat = comments.error_pattern.unwrap();
    assert_eq!(format!("{:?}", pat.0), r#"SubString("foomp")"#);
    assert_eq!(pat.1, 2);
}

#[test]
fn parse_regex_error_pattern() {
    let s = r"
//@  error-pattern:  /foomp/
use std::mem;

    ";
    let comments = Comments::parse(Path::new("<dummy>"), s).unwrap();
    println!("parsed comments: {:#?}", comments);
    let pat = comments.error_pattern.unwrap();
    assert_eq!(format!("{:?}", pat.0), r#"Regex(foomp)"#);
    assert_eq!(pat.1, 2);
}

#[test]
fn parse_slash_slash_at_fail() {
    let s = r"
//@  error-patttern  foomp
use std::mem;

    ";
    assert!(Comments::parse(Path::new("<dummy>"), s).is_err(), "expected parsing to fail");
}

#[test]
fn missing_colon_fail() {
    let s = r"
//@stderr-per-bitwidth hello
use std::mem;

    ";
    assert!(Comments::parse(Path::new("<dummy>"), s).is_err(), "expected parsing to fail");
}
