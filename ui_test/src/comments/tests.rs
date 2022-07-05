use std::path::Path;

use super::Comments;

#[test]
fn parse_simple_comment() {
    let s = r"
use std::mem;

fn main() {
    let _x: &i32 = unsafe { mem::transmute(16usize) }; //~ ERROR encountered a dangling reference (address $HEX is unallocated)
}
    ";
    let comments = Comments::parse(Path::new("<dummy>"), s);
    println!("parsed comments: {:#?}", comments);
    assert_eq!(comments.error_matches[0].definition_line, 5);
    assert_eq!(comments.error_matches[0].revision, None);
    assert_eq!(
        comments.error_matches[0].matched,
        "encountered a dangling reference (address $HEX is unallocated)"
    );
}

#[test]
fn parse_slash_slash_at() {
    let s = r"
//@  error-pattern:  foomp
use std::mem;

    ";
    let comments = Comments::parse(Path::new("<dummy>"), s);
    println!("parsed comments: {:#?}", comments);
    assert_eq!(comments.error_pattern, Some(("foomp".to_string(), 2)));
}

#[test]
#[should_panic]
fn parse_slash_slash_at_fail() {
    let s = r"
//@  error-pattern  foomp
use std::mem;

    ";
    let comments = Comments::parse(Path::new("<dummy>"), s);
    println!("parsed comments: {:#?}", comments);
    assert_eq!(comments.error_pattern, Some(("foomp".to_string(), 2)));
}
