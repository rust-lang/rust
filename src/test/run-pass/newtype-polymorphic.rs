// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct myvec<X>(~[X]);

fn myvec_deref<X:Copy>(mv: myvec<X>) -> ~[X] { return copy *mv; }

fn myvec_elt<X:Copy>(mv: myvec<X>) -> X { return mv[0]; }

pub fn main() {
    let mv = myvec(~[1, 2, 3]);
    assert!((myvec_deref(copy mv)[1] == 2));
    assert!((myvec_elt(copy mv) == 1));
    assert!((mv[2] == 3));
}
