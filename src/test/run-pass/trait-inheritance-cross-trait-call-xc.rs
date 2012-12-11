// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-fast
// aux-build:trait_inheritance_cross_trait_call_xc_aux.rs

extern mod aux(name = "trait_inheritance_cross_trait_call_xc_aux");

trait Bar : aux::Foo {
    fn g() -> int;
}

impl aux::A : Bar {
    fn g() -> int { self.f() }
}

fn main() {
    let a = &aux::A { x: 3 };
    assert a.g() == 10;
}

