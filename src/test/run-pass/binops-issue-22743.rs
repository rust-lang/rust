// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::ops::Mul;

#[derive(Copy, Clone)]
pub struct Foo {
    x: f64,
}

impl Mul<Foo> for f64 {
    type Output = Foo;

    fn mul(self, rhs: Foo) -> Foo {
        // intentionally do something that is not *
        Foo { x: self + rhs.x }
    }
}

pub fn main() {
    let f: Foo = Foo { x: 5.0 };
    let val: f64 = 3.0;
    let f2: Foo = val * f;
    assert_eq!(f2.x, 8.0);
}
