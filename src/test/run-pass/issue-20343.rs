// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Regression test for Issue #20343.

#![deny(dead_code)]

struct B { b: u32 }
struct C;
struct D;

trait T<A> { fn dummy(&self, a: A) { } }
impl<A> T<A> for () {}

impl B {
    // test for unused code in arguments
    fn foo(B { b }: B) -> u32 { b }

    // test for unused code in return type
    fn bar() -> C { unsafe { ::std::mem::transmute(()) } }

    // test for unused code in generics
    fn baz<A: T<D>>() {}
}

pub fn main() {
    let b = B { b: 3 };
    B::foo(b);
    B::bar();
    B::baz::<()>();
}
