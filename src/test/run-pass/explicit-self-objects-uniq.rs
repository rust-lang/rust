// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


trait Foo {
    fn f(self: Box<Self>);
}

struct S {
    x: int
}

impl Foo for S {
    fn f(self: Box<S>) {
        assert_eq!(self.x, 3);
    }
}

pub fn main() {
    let x = box S { x: 3 };
    let y = x as Box<Foo>;
    y.f();
}
