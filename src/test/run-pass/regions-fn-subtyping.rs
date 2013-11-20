// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Issue #2263.

#[allow(dead_assignment)];
#[allow(unused_variable)];

// Should pass region checking.
fn ok(f: |x: &uint|) {
    // Here, g is a function that can accept a uint pointer with
    // lifetime r, and f is a function that can accept a uint pointer
    // with any lifetime.  The assignment g = f should be OK (i.e.,
    // f's type should be a subtype of g's type), because f can be
    // used in any context that expects g's type.  But this currently
    // fails.
    let mut g: <'r>|y: &'r uint| = |x| { };
    g = f;
}

// This version is the same as above, except that here, g's type is
// inferred.
fn ok_inferred(f: |x: &uint|) {
    let mut g: <'r>|x: &'r uint| = |_| {};
    g = f;
}

pub fn main() {
}
