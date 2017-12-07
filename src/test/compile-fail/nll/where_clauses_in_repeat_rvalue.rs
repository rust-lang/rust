// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -Z borrowck=mir -Z nll

#![allow(warnings)]

struct Foo<T> {
    t: T,
}

impl<T: 'static + Copy> Copy for Foo<T> {}
impl<T: 'static + Copy> Clone for Foo<T> {
    fn clone(&self) -> Self {
        *self
    }
}

fn main() {
    let mut x = 22;

    {
        let p = &x;
        //~^ ERROR `x` does not live long enough
        let w = Foo { t: p };

        let v = [w; 22];
    }

    x += 1;
    //~^ ERROR cannot assign to `x` because it is borrowed [E0506]
}
