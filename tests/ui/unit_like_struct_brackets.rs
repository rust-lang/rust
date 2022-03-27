// run-rustfix
#![warn(clippy::unit_like_struct_brackets)]
#![allow(dead_code)]

pub struct MyEmptyStruct {} // should trigger lint
struct MyEmptyTupleStruct(); // should trigger lint

struct MyStruct {
    // should not trigger lint
    field: u8,
}
struct MyTupleStruct(usize, String); // should not trigger lint

fn main() {}
