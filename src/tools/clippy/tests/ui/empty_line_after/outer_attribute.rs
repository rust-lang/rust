//@aux-build:../auxiliary/proc_macro_attr.rs
#![warn(clippy::empty_line_after_outer_attr, clippy::empty_line_after_doc_comments)]

//~v empty_line_after_outer_attr
#[crate_type = "lib"]

fn first_in_crate() {}

#[macro_use]
extern crate proc_macro_attr;

//~v empty_line_after_outer_attr
#[inline]

/// some comment
fn with_one_newline_and_comment() {}

#[inline]
/// some comment
fn with_no_newline_and_comment() {}

//~v empty_line_after_outer_attr
#[inline]

fn with_one_newline() {}

#[rustfmt::skip]
mod two_lines {
    //~v empty_line_after_outer_attr
    #[crate_type = "lib"]


    fn with_two_newlines() {}
}

//~v empty_line_after_outer_attr
#[doc = "doc attributes should be considered attributes"]

enum Baz {
    One,
    Two,
}

//~v empty_line_after_outer_attr
#[repr(C)]

struct Foo {
    one: isize,
    two: isize,
}

//~v empty_line_after_outer_attr
#[allow(dead_code)]

mod foo {}

//~v empty_line_after_outer_attr
#[inline]
// Still lint cases where the empty line does not immediately follow the attribute

fn comment_before_empty_line() {}

//~v empty_line_after_outer_attr
#[allow(unused)]

// This comment is isolated

pub fn isolated_comment() {}

#[doc = "
Returns the escaped value of the textual representation of

"]
pub fn function() -> bool {
    true
}

#[derive(Clone, Copy)]
pub enum FooFighter {
    Bar1,

    Bar2,

    Bar3,

    Bar4,
}

#[crate_type = "lib"]
/*

*/
pub struct EmptyLineInBlockComment;

#[crate_type = "lib"]
/* test */
pub struct BlockComment;

// See https://github.com/rust-lang/rust-clippy/issues/5567
#[rustfmt::skip]
#[fake_async_trait]
pub trait Bazz {
    fn foo() -> Vec<u8> {
        let _i = "";



        vec![]
    }
}

#[derive(Clone, Copy)]
#[dummy(string = "first line

second line
")]
pub struct Args;

fn main() {}
