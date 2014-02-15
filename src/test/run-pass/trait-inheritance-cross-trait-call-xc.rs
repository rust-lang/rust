// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:trait_inheritance_cross_trait_call_xc_aux.rs

extern crate aux = "trait_inheritance_cross_trait_call_xc_aux";

use aux::Foo;

trait Bar : Foo {
    fn g(&self) -> int;
}

impl Bar for aux::A {
    fn g(&self) -> int { self.f() }
}

pub fn main() {
    let a = &aux::A { x: 3 };
    assert_eq!(a.g(), 10);
}
