// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:trait_default_method_xc_aux.rs
// aux-build:trait_default_method_xc_aux_2.rs


extern crate aux = "trait_default_method_xc_aux";
extern crate aux2 = "trait_default_method_xc_aux_2";
use aux::A;
use aux2::{a_struct, welp};


pub fn main () {

    let a = a_struct { x: 0i };
    let b = a_struct { x: 1i };

    assert_eq!(0i.g(), 10);
    assert_eq!(a.g(), 10);
    assert_eq!(a.h(), 11);
    assert_eq!(b.g(), 10);
    assert_eq!(b.h(), 11);
    assert_eq!(A::lurr(&a, &b), 21);

    welp(&0i);
}
