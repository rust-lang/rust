// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that impls on projected self types can resolve overlap, even when the
// projections involve specialization, so long as the associated type is
// provided by the most specialized impl.

#![feature(specialization)]

trait Assoc {
    type Output;
}

default impl<T> Assoc for T {
    type Output = bool;
}

impl Assoc for u8 { type Output = u8; }
impl Assoc for u16 { type Output = u16; }

trait Foo {}
impl Foo for u32 {}
impl Foo for <u8 as Assoc>::Output {}
impl Foo for <u16 as Assoc>::Output {}

fn main() {}
