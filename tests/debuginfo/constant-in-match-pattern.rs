//@ compile-flags:-g
//@ disable-gdb-pretty-printers

#![allow(dead_code, unused_variables)]

// This test makes sure that the compiler doesn't crash when trying to assign
// debug locations to 'constant' patterns in match expressions.

const CONSTANT: u64 = 3;

#[derive(PartialEq, Eq)]
struct Struct {
    a: isize,
    b: usize,
}
const STRUCT: Struct = Struct { a: 1, b: 2 };

#[derive(PartialEq, Eq)]
struct TupleStruct(u32);
const TUPLE_STRUCT: TupleStruct = TupleStruct(4);

#[derive(PartialEq, Eq)]
enum Enum {
    Variant1(char),
    Variant2 { a: u8 },
    Variant3
}
const VARIANT1: Enum = Enum::Variant1('v');
const VARIANT2: Enum = Enum::Variant2 { a: 2 };
const VARIANT3: Enum = Enum::Variant3;

const STRING: &'static str = "String";

fn main() {

    match 1 {
        CONSTANT => {}
        _ => {}
    };

    // if let 3 = CONSTANT {}

    match (Struct { a: 2, b: 2 }) {
        STRUCT => {}
        _ => {}
    };

    // if let STRUCT = STRUCT {}

    match TupleStruct(3) {
        TUPLE_STRUCT => {}
        _ => {}
    };

    // if let TupleStruct(4) = TUPLE_STRUCT {}

    match VARIANT3 {
        VARIANT1 => {},
        VARIANT2 => {},
        VARIANT3 => {},
        _ => {}
    };

    match (VARIANT3, VARIANT2) {
        (VARIANT1, VARIANT3) => {},
        (VARIANT2, VARIANT2) => {},
        (VARIANT3, VARIANT1) => {},
        _ => {}
    };

    // if let VARIANT1 = Enum::Variant3 {}
    // if let VARIANT2 = Enum::Variant3 {}
    // if let VARIANT3 = Enum::Variant3 {}

    match "abc" {
        STRING => {},
        _ => {}
    }

    if let STRING = "def" {}
}
