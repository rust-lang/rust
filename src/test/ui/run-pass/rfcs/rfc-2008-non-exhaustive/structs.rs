// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:structs.rs
extern crate structs;

use structs::{NormalStruct, UnitStruct, TupleStruct};

// We only test matching here as we cannot create non-exhaustive
// structs from another crate. ie. they'll never pass in run-pass tests.

fn match_structs(ns: NormalStruct, ts: TupleStruct, us: UnitStruct) {
    let NormalStruct { first_field, second_field, .. } = ns;

    let TupleStruct { 0: first, 1: second, .. } = ts;

    let UnitStruct { .. } = us;
}

fn main() { }
