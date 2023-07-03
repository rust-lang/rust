//@aux-build:proc_macro_attr.rs:proc-macro
#![warn(clippy::empty_line_after_outer_attr)]
#![allow(clippy::assertions_on_constants)]
#![feature(custom_inner_attributes)]
#![rustfmt::skip]

#[macro_use]
extern crate proc_macro_attr;

// This should produce a warning
#[crate_type = "lib"]

/// some comment
fn with_one_newline_and_comment() { assert!(true) }

// This should not produce a warning
#[crate_type = "lib"]
/// some comment
fn with_no_newline_and_comment() { assert!(true) }


// This should produce a warning
#[crate_type = "lib"]

fn with_one_newline() { assert!(true) }

// This should produce a warning, too
#[crate_type = "lib"]


fn with_two_newlines() { assert!(true) }


// This should produce a warning
#[crate_type = "lib"]

enum Baz {
    One,
    Two
}

// This should produce a warning
#[crate_type = "lib"]

struct Foo {
    one: isize,
    two: isize
}

// This should produce a warning
#[crate_type = "lib"]

mod foo {
}

/// This doc comment should not produce a warning

/** This is also a doc comment and should not produce a warning
 */

// This should not produce a warning
#[allow(non_camel_case_types)]
#[allow(missing_docs)]
#[allow(missing_docs)]
fn three_attributes() { assert!(true) }

// This should not produce a warning
#[doc = "
Returns the escaped value of the textual representation of

"]
pub fn function() -> bool {
    true
}

// This should not produce a warning
#[derive(Clone, Copy)]
pub enum FooFighter {
    Bar1,

    Bar2,

    Bar3,

    Bar4
}

// This should not produce a warning because the empty line is inside a block comment
#[crate_type = "lib"]
/*

*/
pub struct S;

// This should not produce a warning
#[crate_type = "lib"]
/* test */
pub struct T;

// This should not produce a warning
// See https://github.com/rust-lang/rust-clippy/issues/5567
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
