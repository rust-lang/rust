//@ compile-flags:-g

// gdb-command:set print union on
// gdb-command:run

// gdb-command:print case1
// gdb-check:$1 = generic_struct_style_enum::Regular<u16, u32, i64>::Case1{a: 0, b: 31868, c: 31868, d: 31868, e: 31868}

// gdb-command:print case2
// gdb-check:$2 = generic_struct_style_enum::Regular<i16, u32, i64>::Case2{a: 0, b: 286331153, c: 286331153}

// gdb-command:print case3
// gdb-check:$3 = generic_struct_style_enum::Regular<u16, i32, u64>::Case3{a: 0, b: 6438275382588823897}

// gdb-command:print univariant
// gdb-check:$4 = generic_struct_style_enum::Univariant<i32>::TheOnlyCase{a: -1}


#![feature(omit_gdb_pretty_printer_section)]
#![omit_gdb_pretty_printer_section]

use self::Regular::{Case1, Case2, Case3};
use self::Univariant::TheOnlyCase;

// NOTE: This is a copy of the non-generic test case. The `Txx` type parameters have to be
// substituted with something of size `xx` bits and the same alignment as an integer type of the
// same size.

// The first element is to ensure proper alignment, irrespective of the machines word size. Since
// the size of the discriminant value is machine dependent, this has be taken into account when
// datatype layout should be predictable as in this case.
enum Regular<T16, T32, T64> {
    Case1 { a: T64, b: T16, c: T16, d: T16, e: T16},
    Case2 { a: T64, b: T32, c: T32},
    Case3 { a: T64, b: T64 }
}

enum Univariant<T> {
    TheOnlyCase { a: T }
}

fn main() {

    // In order to avoid endianness trouble all of the following test values consist of a single
    // repeated byte. This way each interpretation of the union should look the same, no matter if
    // this is a big or little endian machine.

    // 0b0111110001111100011111000111110001111100011111000111110001111100 = 8970181431921507452
    // 0b01111100011111000111110001111100 = 2088533116
    // 0b0111110001111100 = 31868
    // 0b01111100 = 124
    let case1: Regular<u16, u32, i64> = Case1 { a: 0, b: 31868, c: 31868, d: 31868, e: 31868 };

    // 0b0001000100010001000100010001000100010001000100010001000100010001 = 1229782938247303441
    // 0b00010001000100010001000100010001 = 286331153
    // 0b0001000100010001 = 4369
    // 0b00010001 = 17
    let case2: Regular<i16, u32, i64>  = Case2 { a: 0, b: 286331153, c: 286331153 };

    // 0b0101100101011001010110010101100101011001010110010101100101011001 = 6438275382588823897
    // 0b01011001010110010101100101011001 = 1499027801
    // 0b0101100101011001 = 22873
    // 0b01011001 = 89
    let case3: Regular<u16, i32, u64>  = Case3 { a: 0, b: 6438275382588823897 };

    let univariant = TheOnlyCase { a: -1 };

    zzz(); // #break
}

fn zzz() {()}
