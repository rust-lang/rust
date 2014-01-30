// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub trait OpInt<'a> { fn call<'a>(&'a self, int, int) -> int; }

impl<'a> OpInt<'a> for 'a |int, int| -> int {
    fn call(&self, a:int, b:int) -> int {
        (*self)(a, b)
    }
}

fn squarei<'a>(x: int, op: &'a OpInt) -> int { op.call(x, x) }

fn muli(x:int, y:int) -> int { x * y }

pub fn main() {
    let f = |x,y| muli(x,y);
    {
        let g = &f;
        let h = g as &OpInt;
        squarei(3, h);
    }
}
