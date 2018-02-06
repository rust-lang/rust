// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:issue13507.rs

extern crate issue13507;
use issue13507::testtypes;

use std::any::TypeId;

pub fn type_ids() -> Vec<TypeId> {
    use issue13507::testtypes::*;
    vec![
        TypeId::of::<FooBool>(),
        TypeId::of::<FooInt>(),
        TypeId::of::<FooUint>(),
        TypeId::of::<FooFloat>(),
        TypeId::of::<FooStr>(),
        TypeId::of::<FooArray>(),
        TypeId::of::<FooSlice>(),
        TypeId::of::<FooBox>(),
        TypeId::of::<FooPtr>(),
        TypeId::of::<FooRef>(),
        TypeId::of::<FooFnPtr>(),
        TypeId::of::<FooNil>(),
        TypeId::of::<FooTuple>(),
        TypeId::of::<FooTrait>(),
        TypeId::of::<FooStruct>(),
        TypeId::of::<FooEnum>()
    ]
}

pub fn main() {
    let othercrate = issue13507::testtypes::type_ids();
    let thiscrate = type_ids();
    assert_eq!(thiscrate, othercrate);
}
