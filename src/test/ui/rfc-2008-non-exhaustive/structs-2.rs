// aux-build:structs.rs
extern crate structs;

use structs::{NormalStruct, UnitStruct, TupleStruct, FunctionalRecord};

fn main() {
    let ts_explicit = structs::TupleStruct(640, 480);
    //~^ ERROR struct `structs::TupleStruct` is private
}
