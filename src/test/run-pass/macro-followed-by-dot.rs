// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(macro_rules, tuple_indexing)]

struct Foo(bool);

impl Foo {
    fn frob(&mut self) {
        self.0 = true;
    }
}

impl Drop for Foo {
    fn drop(&mut self) {
        if !self.0 {
            fail!("frob() was not run!")
        }
    }
}

macro_rules! foo {
    ($x: ident) => {{ $x = Foo(false); $x }}
}

fn main() {
    let mut x;
    // This both tests that it is parsed properly AND that it actually runs the method
    foo!(x).frob();
    // Test that other operators can follow the macro invocation
    let mut y;
    foo!(y).0 = true;
}
