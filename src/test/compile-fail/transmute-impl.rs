// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Tests that are conservative around thin/fat pointer mismatches.

#![allow(dead_code)]

use std::mem::transmute;

struct Foo<Sized? T> {
    t: Box<T>
}

impl<Sized? T> Foo<T> {
    fn m(x: &T) -> &int where T : Sized {
        // OK here, because T : Sized is in scope.
        unsafe { transmute(x) }
    }

    fn n(x: &T) -> &int {
        // Not OK here, because T : Sized is not in scope.
        unsafe { transmute(x) } //~ ERROR transmute called on types with potentially different sizes
    }
}

fn main() { }
