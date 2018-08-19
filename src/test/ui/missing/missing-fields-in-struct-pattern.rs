// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct S(usize, usize, usize, usize);

fn main() {
    if let S { a, b, c, d } = S(1, 2, 3, 4) {
    //~^ ERROR struct `S` does not have fields named `a`, `b`, `c`, `d` [E0026]
    //~| ERROR pattern does not mention fields `0`, `1`, `2`, `3` [E0027]
        println!("hi");
    }
}
