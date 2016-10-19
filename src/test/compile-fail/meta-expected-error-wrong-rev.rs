// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// revisions: a
// should-fail

// This is a "meta-test" of the compilertest framework itself.  In
// particular, it includes the right error message, but the message
// targets the wrong revision, so we expect the execution to fail.
// See also `meta-expected-error-correct-rev.rs`.

#[cfg(a)]
fn foo() {
    let x: u32 = 22_usize; //[b]~ ERROR mismatched types
}

fn main() { }
