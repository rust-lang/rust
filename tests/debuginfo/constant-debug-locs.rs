//@ compile-flags:-g

#![allow(dead_code, unused_variables)]
#![feature(omit_gdb_pretty_printer_section)]
#![omit_gdb_pretty_printer_section]

// This test ensures that the compiler doesn't crash when assigning
// debug locations to const expressions. It covers a variety of types 
// and operations to check robustness.

use std::cell::UnsafeCell;

const CONSTANT: u64 = 3 + 4;
const NEGATIVE: i32 = -5;
const FLOAT_CONST: f64 = 1.2 + 3.4;
const BOOLEAN: bool = true && false;

struct Struct {
    a: isize,
    b: usize,
}
const STRUCT: Struct = Struct { a: 1, b: 2 };

struct TupleStruct(u32);
const TUPLE_STRUCT: TupleStruct = TupleStruct(4);

enum Enum {
    Variant1(char),
    Variant2 { a: u8 },
    Variant3
}

const VARIANT1: Enum = Enum::Variant1('v');
const VARIANT2: Enum = Enum::Variant2 { a: 2 };
const VARIANT3: Enum = Enum::Variant3;

const STRING: &'static str = "String";

const VEC: [u32; 8] = [0; 8];
const TUPLE: (i32, bool) = (42, true);
const ARRAY: [i32; 3] = [1, 2, 3];
const REFERENCE: &str = &STRING;

const NESTED: (Struct, TupleStruct) = (STRUCT, TUPLE_STRUCT);

const UNSAFE_CELL: UnsafeCell<bool> = UnsafeCell::new(false);

fn main() {
    let mut _constant = CONSTANT;
    let mut _negative = NEGATIVE;
    let mut _float_const = FLOAT_CONST;
    let mut _boolean = BOOLEAN;
    let mut _struct = STRUCT;
    let mut _tuple_struct = TUPLE_STRUCT;
    let mut _variant1 = VARIANT1;
    let mut _variant2 = VARIANT2;
    let mut _variant3 = VARIANT3;
    let mut _string = STRING;
    let mut _vec = VEC;
    let mut _tuple = TUPLE;
    let mut _array = ARRAY;
    let mut _reference = REFERENCE;
    let mut _nested = NESTED;
    let mut _unsafe_cell = UNSAFE_CELL;

    assert_eq!(CONSTANT, 7);
    assert_eq!(NEGATIVE, -5);
    assert_eq!(FLOAT_CONST, 4.6);
    assert_eq!(BOOLEAN, false);
    assert_eq!(STRUCT.a, 1);
    assert_eq!(STRUCT.b, 2);
    assert_eq!(TUPLE_STRUCT.0, 4);
    if let Enum::Variant1(c) = VARIANT1 {
        assert_eq!(c, 'v');
    }
    assert_eq!(STRING, "String");
    assert_eq!(VEC, [0; 8]);
    assert_eq!(TUPLE, (42, true));
    assert_eq!(ARRAY, [1, 2, 3]);
    assert_eq!(REFERENCE, &"String");
}
