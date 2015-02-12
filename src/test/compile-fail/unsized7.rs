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

use std::marker::MarkerTrait;

trait T : MarkerTrait {}

// I would like these to fail eventually.
// impl - bounded
trait T1<Z: T> {
    fn dummy(&self) -> Z;
}

struct S3<Y: ?Sized>(Box<Y>);
impl<X: ?Sized + T> T1<X> for S3<X> {
    //~^ ERROR `core::marker::Sized` is not implemented for the type `X`
}

fn main() { }
