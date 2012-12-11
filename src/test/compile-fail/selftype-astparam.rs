// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait add {
    fn plus(++x: self) -> self;
}

impl int: add {
    fn plus(++x: int) -> int { self + x }
}

fn do_add<A:add>(x: A, y: A) -> A { x.plus(y) }

fn main() {
    let x = 3 as add;
    let y = 4 as add;
    do_add(x, y); //~ ERROR a boxed trait with self types may not be passed as a bounded type
}
