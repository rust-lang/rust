// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// pretty-expanded FIXME #23616

pub trait OpInt { fn call(&mut self, isize, isize) -> isize; }

impl<F> OpInt for F where F: FnMut(isize, isize) -> isize {
    fn call(&mut self, a:isize, b:isize) -> isize {
        (*self)(a, b)
    }
}

fn squarei<'a>(x: isize, op: &'a mut OpInt) -> isize { op.call(x, x) }

fn muli(x:isize, y:isize) -> isize { x * y }

pub fn main() {
    let mut f = |x, y| muli(x, y);
    {
        let g = &mut f;
        let h = g as &mut OpInt;
        squarei(3, h);
    }
}
