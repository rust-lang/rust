// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that we select between traits A and B. To do that, we must
// consider the `Sized` bound.

trait A {
    fn foo(self);
}

trait B {
    fn foo(self);
}

impl<T: Sized> A for *const T {
    fn foo(self) {}
}

impl<T> B for *const [T] {
    fn foo(self) {}
}

fn main() {
    let x: [int, ..4] = [1,2,3,4];
    let xptr = x.as_slice() as *const _;
    xptr.foo();
}
