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

// pretty-expanded FIXME #23616

#![allow(dead_assignment)]
#![allow(unused_variables)]
#![allow(unknown_features)]

// FIXME (#22405): Replace `Box::new` with `box` here when/if possible.

// Should pass region checking.
fn ok(f: Box<FnMut(&usize)>) {
    // Here, g is a function that can accept a usize pointer with
    // lifetime r, and f is a function that can accept a usize pointer
    // with any lifetime.  The assignment g = f should be OK (i.e.,
    // f's type should be a subtype of g's type), because f can be
    // used in any context that expects g's type.  But this currently
    // fails.
    let mut g: Box<for<'r> FnMut(&'r usize)> = Box::new(|x| { });
    g = f;
}

// This version is the same as above, except that here, g's type is
// inferred.
fn ok_inferred(f: Box<FnMut(&usize)>) {
    let mut g: Box<for<'r> FnMut(&'r usize)> = Box::new(|_| {});
    g = f;
}

pub fn main() {
}
