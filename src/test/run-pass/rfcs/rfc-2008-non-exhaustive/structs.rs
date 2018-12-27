// run-pass
#![allow(dead_code)]
#![allow(unused_variables)]
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
