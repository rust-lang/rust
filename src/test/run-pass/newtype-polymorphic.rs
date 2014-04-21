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
struct myvec<X>(Vec<X> );

fn myvec_deref<X:Clone>(mv: myvec<X>) -> Vec<X> {
    let myvec(v) = mv;
    return v.clone();
}

fn myvec_elt<X>(mv: myvec<X>) -> X {
    let myvec(v) = mv;
    return v.move_iter().next().unwrap();
}

pub fn main() {
    let mv = myvec(vec!(1i, 2, 3));
    let mv_clone = mv.clone();
    let mv_clone = myvec_deref(mv_clone);
    assert_eq!(*mv_clone.get(1), 2);
    assert_eq!(myvec_elt(mv.clone()), 1);
    let myvec(v) = mv;
    assert_eq!(*v.get(2), 3);
}
