// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

enum maybe_pointy {
    none,
    p(@pointy),
}

type pointy = {
    mut a : maybe_pointy,
    d : fn~() -> uint,
};

fn make_uniq_closure<A:Owned Copy>(a: A) -> fn~() -> uint {
    fn~() -> uint { ptr::addr_of(&a) as uint }
}

fn empty_pointy() -> @pointy {
    return @{
        mut a : none,
        d : make_uniq_closure(~"hi")
    }
}

fn main()
{
    let v = empty_pointy();
    v.a = p(v);
}
