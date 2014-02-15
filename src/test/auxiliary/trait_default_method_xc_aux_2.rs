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

extern crate aux = "trait_default_method_xc_aux";
use aux::A;

pub struct a_struct { x: int }

impl A for a_struct {
    fn f(&self) -> int { 10 }
}

// This function will need to get inlined, and badness may result.
pub fn welp<A>(x: A) -> A {
    let a = a_struct { x: 0 };
    a.g();
    x
}
