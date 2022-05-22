// aux-build:structs.rs
extern crate structs;

use structs::{NormalStruct, UnitStruct, TupleStruct, FunctionalRecord};

fn main() {
    let fr = FunctionalRecord {
    //~^ ERROR cannot create non-exhaustive struct
        first_field: 1920,
        second_field: 1080,
        ..FunctionalRecord::default()
    };

    let ns = NormalStruct { first_field: 640, second_field: 480 };
    //~^ ERROR cannot create non-exhaustive struct

    let NormalStruct { first_field, second_field } = ns;
    //~^ ERROR `..` required with struct marked as non-exhaustive

    let ts = TupleStruct(640, 480);
    //~^ ERROR cannot initialize a tuple struct which contains private fields [E0423]

    let ts_explicit = structs::TupleStruct(640, 480);
    //~^ ERROR tuple struct constructor `TupleStruct` is private [E0603]

    let TupleStruct { 0: first_field, 1: second_field } = ts;
    //~^ ERROR `..` required with struct marked as non-exhaustive

    let us = UnitStruct;
    //~^ ERROR expected value, found struct `UnitStruct` [E0423]

    let us_explicit = structs::UnitStruct;
    //~^ ERROR unit struct `UnitStruct` is private [E0603]

    let UnitStruct { } = us;
    //~^ ERROR `..` required with struct marked as non-exhaustive
}

// Everything below this is expected to compile successfully.

// We only test matching here as we cannot create non-exhaustive
// structs from another crate. ie. they'll never pass in run-pass tests.
fn match_structs(ns: NormalStruct, ts: TupleStruct, us: UnitStruct) {
    let NormalStruct { first_field, second_field, .. } = ns;

    let TupleStruct { 0: first, 1: second, .. } = ts;

    let UnitStruct { .. } = us;
}
