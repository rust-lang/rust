// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// The type of `y` ends up getting inferred to the type of the block.
fn broken() -> int {
    let mut x = 3;
    let mut y = ~[&mut x];
    while x < 10 {
        let mut z = x;
        y += ~[&mut z]; //~ ERROR illegal borrow
        x += 1;
    }
    vec::foldl(0, y, |v, p| v + **p )
}

fn main() { }