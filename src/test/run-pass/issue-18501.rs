// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that we don't ICE when inlining a function from another
// crate that uses a trait method as a value due to incorrectly
// translating the def ID of the trait during AST decoding.

// aux-build:issue-18501.rs
// pretty-expanded FIXME #23616

extern crate issue_18501 as issue;

fn main() {
    issue::pass_method();
}
