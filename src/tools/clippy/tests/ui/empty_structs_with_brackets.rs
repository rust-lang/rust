//@run-rustfix
#![warn(clippy::empty_structs_with_brackets)]
#![allow(dead_code)]

pub struct MyEmptyStruct {} // should trigger lint
struct MyEmptyTupleStruct(); // should trigger lint

// should not trigger lint
struct MyCfgStruct {
    #[cfg(feature = "thisisneverenabled")]
    field: u8,
}

// should not trigger lint
struct MyCfgTupleStruct(#[cfg(feature = "thisisneverenabled")] u8);

// should not trigger lint
struct MyStruct {
    field: u8,
}
struct MyTupleStruct(usize, String); // should not trigger lint
struct MySingleTupleStruct(usize); // should not trigger lint
struct MyUnitLikeStruct; // should not trigger lint

fn main() {}
