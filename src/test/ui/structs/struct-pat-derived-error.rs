// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct A {
    b: usize,
    c: usize
}

impl A {
    fn foo(&self) {
        let A { x, y } = self.d; //~ ERROR no field `d` on type `&A`
        //~^ ERROR struct `A` does not have fields named `x`, `y`
        //~| ERROR pattern does not mention fields `b`, `c`
    }
}

fn main() {}
