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
    a: isize
}

trait Changer {
    fn change(self) -> Self;
}

impl Changer for X {
    fn change(mut self) -> X {
        self.a = 55;
        self
    }
}

pub fn main() {
    let x = X { a: 32 };
    let new_x = x.change();
    assert_eq!(new_x.a, 55);
}
