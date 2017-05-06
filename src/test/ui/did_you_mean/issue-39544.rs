// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub enum X {
    Y
}

pub struct Z {
    x: X
}

fn main() {
    let z = Z { x: X::Y };
    let _ = &mut z.x;
}

impl Z {
    fn foo<'z>(&'z self) {
        let _ = &mut self.x;
    }

    fn foo1(&self, other: &Z) {
        let _ = &mut self.x;
        let _ = &mut other.x;
    }

    fn foo2<'a>(&'a self, other: &Z) {
        let _ = &mut self.x;
        let _ = &mut other.x;
    }

    fn foo3<'a>(self: &'a Self, other: &Z) {
        let _ = &mut self.x;
        let _ = &mut other.x;
    }

    fn foo4(other: &Z) {
        let _ = &mut other.x;
    }

}

pub fn with_arg(z: Z, w: &Z) {
    let _ = &mut z.x;
    let _ = &mut w.x;
}

pub fn with_tuple() {
    let mut y = 0;
    let x = (&y,);
    *x.0 = 1;
}
