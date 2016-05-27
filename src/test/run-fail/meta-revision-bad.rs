// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Meta test for compiletest: check that when we give the wrong error
// patterns, the test fails.

// revisions: foo bar
// should-fail
//[foo] error-pattern:bar
//[bar] error-pattern:foo

#[cfg(foo)]
fn die() {
    panic!("foo");
}
#[cfg(bar)]
fn die() {
    panic!("bar");
}

fn main() {
    die();
}
