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

use std::intrinsics::TypeId;

pub fn type_ids() -> Vec<TypeId> {
    let mut ids = vec!();
    ids.push(TypeId::of::<testtypes::FooNil>());
    ids.push(TypeId::of::<testtypes::FooBool>());
    ids.push(TypeId::of::<testtypes::FooInt>());
    ids.push(TypeId::of::<testtypes::FooUint>());
    ids.push(TypeId::of::<testtypes::FooFloat>());
    ids.push(TypeId::of::<testtypes::FooEnum>());
    ids.push(TypeId::of::<testtypes::FooUniq>());
    ids.push(TypeId::of::<testtypes::FooPtr>());
    ids.push(TypeId::of::<testtypes::FooClosure>());
    ids.push(TypeId::of::<&'static testtypes::FooTrait>());
    ids.push(TypeId::of::<testtypes::FooStruct>());
    ids.push(TypeId::of::<testtypes::FooTuple>());
    ids
}

pub fn main() {
    let othercrate = testtypes::type_ids();
    let thiscrate = type_ids();
    assert_eq!(thiscrate, othercrate);
}
