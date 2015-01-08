// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Check that we properly record borrows when we are doing an
// overloaded, autoderef of a value obtained via an overloaded index
// operator. The accounting of the all the implicit things going on
// here is rather subtle. Issue #20232.

use std::ops::{Deref, Index};

struct MyVec<T> { x: T }

impl<T> Index<usize> for MyVec<T> {
    type Output = T;
    fn index(&self, _: &usize) -> &T {
        &self.x
    }
}

struct MyPtr<T> { x: T }

impl<T> Deref for MyPtr<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.x
    }
}

struct Foo { f: usize }

fn main() {
    let mut v = MyVec { x: MyPtr { x: Foo { f: 22 } } };
    let i = &v[0].f;
    v = MyVec { x: MyPtr { x: Foo { f: 23 } } };
    //~^ ERROR cannot assign to `v`
    read(*i);
}

fn read(_: usize) { }

