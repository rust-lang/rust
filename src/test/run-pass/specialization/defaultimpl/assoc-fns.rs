// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that non-method associated functions can be specialized

#![feature(specialization)]

trait Foo {
    fn mk() -> Self;
}

default impl<T: Default> Foo for T {
    fn mk() -> T {
        T::default()
    }
}

impl Foo for Vec<u8> {
    fn mk() -> Vec<u8> {
        vec![0]
    }
}

fn main() {
    let v1: Vec<i32> = Foo::mk();
    let v2: Vec<u8> = Foo::mk();

    assert!(v1.len() == 0);
    assert!(v2.len() == 1);
}
