// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
#![feature(arbitrary_self_types)]

use std::rc::Rc;

struct Foo {
    x: i32,
    y: i32,
}

impl Foo {
    fn x(self: &Rc<Self>) -> i32 {
        self.x
    }

    fn y(self: Rc<Self>) -> i32 {
        self.y
    }
}

fn main() {
    let foo = Rc::new(Foo {x: 3, y: 4});
    assert_eq!(3, foo.x());
    assert_eq!(4, foo.y());
}
