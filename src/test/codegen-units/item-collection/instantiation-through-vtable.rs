// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-tidy-linelength
// compile-flags:-Zprint-trans-items=eager

#![deny(dead_code)]

trait Trait {
    fn foo(&self) -> u32;
    fn bar(&self);
}

struct Struct<T> {
    _a: T
}

impl<T> Trait for Struct<T> {
    fn foo(&self) -> u32 { 0 }
    fn bar(&self) {}
}

//~ TRANS_ITEM fn instantiation_through_vtable::main[0]
fn main() {
    let s1 = Struct { _a: 0u32 };

    //~ TRANS_ITEM drop-glue i8
    //~ TRANS_ITEM fn instantiation_through_vtable::{{impl}}[0]::foo[0]<u32>
    //~ TRANS_ITEM fn instantiation_through_vtable::{{impl}}[0]::bar[0]<u32>
    let _ = &s1 as &Trait;

    let s1 = Struct { _a: 0u64 };
    //~ TRANS_ITEM fn instantiation_through_vtable::{{impl}}[0]::foo[0]<u64>
    //~ TRANS_ITEM fn instantiation_through_vtable::{{impl}}[0]::bar[0]<u64>
    let _ = &s1 as &Trait;
}
