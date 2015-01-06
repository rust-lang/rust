// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(unboxed_closures)]

use std::any::Any;
use std::intrinsics::TypeId;

pub trait Pt {}
pub trait Rt {}

trait Private<P: Pt, R: Rt> {
    fn call(&self, p: P, r: R);
}
pub trait Public: Private<
    <Self as Public>::P,
//~^ ERROR illegal recursive type; insert an enum or struct in the cycle, if this is desired
    <Self as Public>::R
> {
    type P;
    type R;

    fn call_inner(&self);
}

fn main() {}
