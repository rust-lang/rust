// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait Foo<T> {
    fn f(&self, x: &T);
}

trait Bar : Sized + Foo<Self> {
    fn g(&self);
}

struct S {
    x: isize
}

impl Foo<S> for S {
    fn f(&self, x: &S) {
        println!("{}", x.x);
    }
}

impl Bar for S {
    fn g(&self) {
        self.f(self);
    }
}

pub fn main() {
    let s = S { x: 1 };
    s.g();
}
