// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test paths to associated types using the type-parameter-only sugar.

pub trait Foo {
    type A;
    fn boo(&self) -> Self::A;
}

impl Foo for int {
    type A = uint;
    fn boo(&self) -> uint {
        5
    }
}

// Using a type via a function.
pub fn bar<T: Foo>(a: T, x: T::A) -> T::A {
    let _: T::A = a.boo();
    x
}

// Using a type via an impl.
trait C {
    fn f();
    fn g(&self) { }
}
struct B<X>(X);
impl<T: Foo> C for B<T> {
    fn f() {
        let x: T::A = panic!();
    }
}

pub fn main() {
    let z: uint = bar(2, 4_usize);
}
