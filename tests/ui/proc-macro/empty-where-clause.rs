//@ proc-macro: test-macros.rs

extern crate test_macros;
use test_macros::recollect_attr;

#[recollect_attr]
struct FieldStruct where {
    field: MissingType1 //~ ERROR cannot find
}

#[recollect_attr]
struct TupleStruct(MissingType2) where; //~ ERROR cannot find

enum MyEnum where {
    Variant(MissingType3) //~ ERROR cannot find
}

fn main() {}
