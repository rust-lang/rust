// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


struct X {
    a: int
}

trait Changer {
    fn change(mut self: Box<Self>) -> Box<Self>;
}

impl Changer for X {
    fn change(mut self: Box<X>) -> Box<X> {
        self.a = 55;
        self
    }
}

pub fn main() {
    let x = box X { a: 32 };
    let new_x = x.change();
    assert_eq!(new_x.a, 55);
}
