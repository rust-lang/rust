// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::ptr;

enum maybe_pointy {
    none,
    p(@mut Pointy),
}

struct Pointy {
    a : maybe_pointy,
    d : ~fn() -> uint,
}

fn make_uniq_closure<A:Send + Copy>(a: A) -> ~fn() -> uint {
    let result: ~fn() -> uint = || ptr::to_unsafe_ptr(&a) as uint;
    result
}

fn empty_pointy() -> @mut Pointy {
    return @mut Pointy {
        a : none,
        d : make_uniq_closure(~"hi")
    }
}

pub fn main() {
    let v = empty_pointy();
    v.a = p(v);
}
