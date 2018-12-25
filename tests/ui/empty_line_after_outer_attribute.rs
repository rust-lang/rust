// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![warn(clippy::empty_line_after_outer_attr)]
#![allow(clippy::assert_checks::explicit_true)]
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

fn main() { }
