// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[deriving(Clone)]
struct myvec<X>(~[X]);

fn myvec_deref<X:Clone>(mv: myvec<X>) -> ~[X] { return (*mv).clone(); }

fn myvec_elt<X>(mv: myvec<X>) -> X { return mv[0]; }

pub fn main() {
    let mv = myvec(~[1, 2, 3]);
    assert_eq!(myvec_deref(mv.clone())[1], 2);
    assert_eq!(myvec_elt(mv.clone()), 1);
    assert_eq!(mv[2], 3);
}
