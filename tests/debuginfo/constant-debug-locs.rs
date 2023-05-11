// min-lldb-version: 310

// compile-flags:-g

#![allow(dead_code, unused_variables)]
#![feature(omit_gdb_pretty_printer_section)]
#![omit_gdb_pretty_printer_section]

// This test makes sure that the compiler doesn't crash when trying to assign
// debug locations to const-expressions.

use std::cell::UnsafeCell;

const CONSTANT: u64 = 3 + 4;

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

const NESTED: (Struct, TupleStruct) = (STRUCT, TUPLE_STRUCT);

const UNSAFE_CELL: UnsafeCell<bool> = UnsafeCell::new(false);

fn main() {
    let mut _constant = CONSTANT;
    let mut _struct = STRUCT;
    let mut _tuple_struct = TUPLE_STRUCT;
    let mut _variant1 = VARIANT1;
    let mut _variant2 = VARIANT2;
    let mut _variant3 = VARIANT3;
    let mut _string = STRING;
    let mut _vec = VEC;
    let mut _nested = NESTED;
    let mut _unsafe_cell = UNSAFE_CELL;
}
