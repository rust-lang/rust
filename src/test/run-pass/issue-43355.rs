// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Check that the code for issue #43355 can run without an ICE, please remove
// this test when it becomes an hard error.

pub trait Trait1<X> {
    type Output;
}
pub trait Trait2<X> {}

impl<X, T> Trait1<X> for T where T: Trait2<X> {
    type Output = ();
}
impl<X> Trait1<Box<X>> for A {
    type Output = i32;
}

pub struct A;

fn f<X, T: Trait1<Box<X>>>() {
    println!("k: {}", ::std::mem::size_of::<<T as Trait1<Box<X>>>::Output>());
}

pub fn g<X, T: Trait2<Box<X>>>() {
    f::<X, T>();
}

fn main() {}
