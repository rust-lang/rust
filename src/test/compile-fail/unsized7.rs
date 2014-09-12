// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test sized-ness checking in substitution in impls.

trait T for Sized? {}

// I would like these to fail eventually.
// impl - bounded
trait T1<Z: T> {
}
struct S3<Sized? Y>;
impl<Sized? X: T> T1<X> for S3<X> {
    //~^ ERROR `core::kinds::Sized` is not implemented for the type `X`
}

// impl - unbounded
trait T2<Z> {
}
struct S4<Sized? Y>;
impl<Sized? X> T2<X> for S4<X> {
    //~^ ERROR `core::kinds::Sized` is not implemented for the type `X`
}

// impl - struct
trait T3<Sized? Z> {
}
struct S5<Y>;
impl<Sized? X> T3<X> for S5<X> { //~ ERROR not implemented
}

impl<Sized? X> S5<X> { //~ ERROR not implemented
}


fn main() { }
