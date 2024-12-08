//@ min-lldb-version: 1800

//@ compile-flags:-g

// === GDB TESTS ===================================================================================

// gdb-command:set print union on
// gdb-command:run

// gdb-command:print case1
// gdb-check:$1 = tuple_style_enum::Regular::Case1(0, 31868, 31868, 31868, 31868)

// gdb-command:print case2
// gdb-check:$2 = tuple_style_enum::Regular::Case2(0, 286331153, 286331153)

// gdb-command:print case3
// gdb-check:$3 = tuple_style_enum::Regular::Case3(0, 6438275382588823897)

// gdb-command:print univariant
// gdb-check:$4 = tuple_style_enum::Univariant::TheOnlyCase(-1)


// === LLDB TESTS ==================================================================================

// lldb-command:run

// lldb-command:v case1
// lldb-check:(tuple_style_enum::Regular) case1 = { value = { 0 = 0 1 = 31868 2 = 31868 3 = 31868 4 = 31868 } $discr$ = 0 }

// lldb-command:v case2
// lldb-check:(tuple_style_enum::Regular) case2 = { value = { 0 = 0 1 = 286331153 2 = 286331153 } $discr$ = 1 }

// lldb-command:v case3
// lldb-check:(tuple_style_enum::Regular) case3 = { value = { 0 = 0 1 = 6438275382588823897 } $discr$ = 2 }

// lldb-command:v univariant
// lldb-check:(tuple_style_enum::Univariant) univariant = { value = { 0 = -1 } }

#![allow(unused_variables)]
#![feature(omit_gdb_pretty_printer_section)]
#![omit_gdb_pretty_printer_section]

use self::Regular::{Case1, Case2, Case3};
use self::Univariant::TheOnlyCase;

// The first element is to ensure proper alignment, irrespective of the machines word size. Since
// the size of the discriminant value is machine dependent, this has be taken into account when
// datatype layout should be predictable as in this case.
enum Regular {
    Case1(u64, u16, u16, u16, u16),
    Case2(u64, u32, u32),
    Case3(u64, u64)
}

enum Univariant {
    TheOnlyCase(i64)
}

fn main() {

    // In order to avoid endianness trouble all of the following test values consist of a single
    // repeated byte. This way each interpretation of the union should look the same, no matter if
    // this is a big or little endian machine.

    // 0b0111110001111100011111000111110001111100011111000111110001111100 = 8970181431921507452
    // 0b01111100011111000111110001111100 = 2088533116
    // 0b0111110001111100 = 31868
    // 0b01111100 = 124
    let case1 = Case1(0, 31868, 31868, 31868, 31868);

    // 0b0001000100010001000100010001000100010001000100010001000100010001 = 1229782938247303441
    // 0b00010001000100010001000100010001 = 286331153
    // 0b0001000100010001 = 4369
    // 0b00010001 = 17
    let case2 = Case2(0, 286331153, 286331153);

    // 0b0101100101011001010110010101100101011001010110010101100101011001 = 6438275382588823897
    // 0b01011001010110010101100101011001 = 1499027801
    // 0b0101100101011001 = 22873
    // 0b01011001 = 89
    let case3 = Case3(0, 6438275382588823897);

    let univariant = TheOnlyCase(-1);

    zzz(); // #break
}

fn zzz() {()}
