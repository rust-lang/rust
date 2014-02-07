// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-fast
// aux-build:trait_inheritance_auto_xc_2_aux.rs

extern mod aux = "trait_inheritance_auto_xc_2_aux";

// aux defines impls of Foo, Bar and Baz for A
use aux::{Foo, Bar, Baz, A};

// We want to extend all Foo, Bar, Bazes to Quuxes
pub trait Quux: Foo + Bar + Baz { }
impl<T:Foo + Bar + Baz> Quux for T { }

fn f<T:Quux>(a: &T) {
    assert_eq!(a.f(), 10);
    assert_eq!(a.g(), 20);
    assert_eq!(a.h(), 30);
}

pub fn main() {
    let a = &A { x: 3 };
    f(a);
}
