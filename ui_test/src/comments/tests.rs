use std::path::Path;

use super::Comments;

use crate::tests::init;
use color_eyre::eyre::{bail, Result};

#[test]
fn parse_simple_comment() -> Result<()> {
    init();
    let s = r"
use std::mem;

fn main() {
    let _x: &i32 = unsafe { mem::transmute(16usize) }; //~ ERROR encountered a dangling reference (address $HEX is unallocated)
}
    ";
    let comments = Comments::parse(Path::new("<dummy>"), s)?;
    println!("parsed comments: {:#?}", comments);
    assert_eq!(comments.error_matches[0].definition_line, 5);
    assert_eq!(comments.error_matches[0].revision, None);
    assert_eq!(
        comments.error_matches[0].matched,
        "encountered a dangling reference (address $HEX is unallocated)"
    );
    Ok(())
}

#[test]
fn parse_slash_slash_at() -> Result<()> {
    init();
    let s = r"
//@  error-pattern:  foomp
use std::mem;

    ";
    let comments = Comments::parse(Path::new("<dummy>"), s)?;
    println!("parsed comments: {:#?}", comments);
    assert_eq!(comments.error_pattern, Some(("foomp".to_string(), 2)));
    Ok(())
}

#[test]
fn parse_slash_slash_at_fail() -> Result<()> {
    init();
    let s = r"
//@  error-patttern  foomp
use std::mem;

    ";
    match Comments::parse(Path::new("<dummy>"), s) {
        Ok(_) => bail!("expected parsing to fail"),
        Err(_) => Ok(()),
    }
}
