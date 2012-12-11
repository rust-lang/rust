// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pure fn range(from: uint, to: uint, f: fn(uint) -> bool) {
    let mut i = from;
    while i < to {
        if !f(i) {return;} // Note: legal to call argument, even if it is not pure.
        i += 1u;
    }
}

pure fn range2(from: uint, to: uint, f: fn(uint)) {
    for range(from, to) |i| {
        f(i*2u);
    }
}

pure fn range3(from: uint, to: uint, f: {x: fn(uint)}) {
    for range(from, to) |i| {
        (f.x)(i*2u); //~ ERROR access to impure function prohibited
    }
}

fn main() {}
