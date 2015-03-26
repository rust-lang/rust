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

struct MyType;

impl MyType {
    const IMPL_IS_INHERENT: bool = true;
}

trait MyTrait: MarkerTrait {
    const IMPL_IS_INHERENT: bool;
    const IMPL_IS_ON_TRAIT: bool;
}

impl MyTrait for MyType {
    const IMPL_IS_INHERENT: bool = false;
    const IMPL_IS_ON_TRAIT: bool = true;
}

fn main() {
    // Check that the inherent impl is used before the trait, but that the trait
    // can still be accessed.
    assert!(<MyType>::IMPL_IS_INHERENT);
    assert!(!<MyType as MyTrait>::IMPL_IS_INHERENT);
    assert!(<MyType>::IMPL_IS_ON_TRAIT);
}
