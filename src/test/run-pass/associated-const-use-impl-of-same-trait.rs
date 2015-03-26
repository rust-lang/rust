// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(associated_consts)]

use std::marker::MarkerTrait;

// The main purpose of this test is to ensure that different impls of the same
// trait can refer to each other without setting off the static recursion check
// (as long as there's no actual recursion).

trait Foo: MarkerTrait {
    const BAR: u32;
}

struct IsFoo1;

impl Foo for IsFoo1 {
    const BAR: u32 = 1;
}

struct IsFoo2;

impl Foo for IsFoo2 {
    const BAR: u32 = <IsFoo1 as Foo>::BAR;
}

fn main() {
    assert_eq!(<IsFoo1>::BAR, <IsFoo2 as Foo>::BAR);
}
