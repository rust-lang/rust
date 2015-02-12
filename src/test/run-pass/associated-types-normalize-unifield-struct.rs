// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Regression test for issue #21010: Normalize associated types in
// various special paths in the `type_is_immediate` function.


pub trait OffsetState: Sized {}
pub trait Offset {
    type State: OffsetState;
    fn dummy(&self) { }
}

#[derive(Copy)] pub struct X;
impl Offset for X { type State = Y; }

#[derive(Copy)] pub struct Y;
impl OffsetState for Y {}

pub fn now() -> DateTime<X> { from_utc(Y) }

pub struct DateTime<Off: Offset> { pub offset: Off::State }
pub fn from_utc<Off: Offset>(offset: Off::State) -> DateTime<Off> { DateTime { offset: offset } }

pub fn main() {
    let _x = now();
}
