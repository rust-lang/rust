// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Some traits can be derived for unions.

#![feature(untagged_unions)]

#[derive(
    Copy,
    Clone,
    Eq,
)]
union U {
    a: u8,
    b: u16,
}

impl PartialEq for U { fn eq(&self, rhs: &Self) -> bool { true } }

#[derive(
    Clone,
    Copy,
    Eq
)]
union W<T> {
    a: T,
}

impl<T> PartialEq for W<T> { fn eq(&self, rhs: &Self) -> bool { true } }

fn main() {
    let u = U { b: 0 };
    let u1 = u;
    let u2 = u.clone();
    assert!(u1 == u2);

    let w = W { a: 0 };
    let w1 = w.clone();
    assert!(w == w1);
}
