// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![crate_type="lib"]

struct Foo;
// This is the ICE trigger
struct Formatter;

trait Show {
    fn fmt(&self);
}

impl Show for Foo {
    fn fmt(&self) {}
}

fn bar<T>(f: extern "Rust" fn(&T), t: &T) { }

// ICE requirement: this has to be marked as inline
#[inline]
pub fn baz() {
    bar(Show::fmt, &Foo);
}
