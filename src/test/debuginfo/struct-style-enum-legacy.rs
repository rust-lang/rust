// ignore-tidy-linelength
// min-lldb-version: 310

// As long as LLVM 5 and LLVM 6 are supported, we want to test the
// enum debuginfo fallback mode.  Once those are desupported, this
// test can be removed, as there is another (non-"legacy") test that
// tests the new mode.
// ignore-llvm-version: 7.0 - 9.9.9
// ignore-gdb-version: 7.11.90 - 7.12.9
// ignore-gdb-version: 8.2 - 9.9

// compile-flags:-g

// === GDB TESTS ===================================================================================

// gdb-command:set print union on
// gdb-command:run

// gdb-command:print case1
// gdbg-check:$1 = {{RUST$ENUM$DISR = Case1, a = 0, b = 31868, c = 31868, d = 31868, e = 31868}, {RUST$ENUM$DISR = Case1, [...]}, {RUST$ENUM$DISR = Case1, [...]}}
// gdbr-check:$1 = struct_style_enum_legacy::Regular::Case1{a: 0, b: 31868, c: 31868, d: 31868, e: 31868}

// gdb-command:print case2
// gdbg-check:$2 = {{RUST$ENUM$DISR = Case2, [...]}, {RUST$ENUM$DISR = Case2, a = 0, b = 286331153, c = 286331153}, {RUST$ENUM$DISR = Case2, [...]}}
// gdbr-check:$2 = struct_style_enum_legacy::Regular::Case2{a: 0, b: 286331153, c: 286331153}

// gdb-command:print case3
// gdbg-check:$3 = {{RUST$ENUM$DISR = Case3, [...]}, {RUST$ENUM$DISR = Case3, [...]}, {RUST$ENUM$DISR = Case3, a = 0, b = 6438275382588823897}}
// gdbr-check:$3 = struct_style_enum_legacy::Regular::Case3{a: 0, b: 6438275382588823897}

// gdb-command:print univariant
// gdbg-check:$4 = {{a = -1}}
// gdbr-check:$4 = struct_style_enum_legacy::Univariant::TheOnlyCase{a: -1}


// === LLDB TESTS ==================================================================================

// lldb-command:run

// lldb-command:print case1
// lldbg-check:[...]$0 = Case1 { a: 0, b: 31868, c: 31868, d: 31868, e: 31868 }
// lldbr-check:(struct_style_enum_legacy::Regular::Case1) case1 = { a = 0 b = 31868 c = 31868 d = 31868 e = 31868 }

// lldb-command:print case2
// lldbg-check:[...]$1 = Case2 { a: 0, b: 286331153, c: 286331153 }
// lldbr-check:(struct_style_enum_legacy::Regular::Case2) case2 = Case2 { struct_style_enum_legacy::Regular::Case1: 0, struct_style_enum_legacy::Regular::Case2: 286331153, struct_style_enum_legacy::Regular::Case3: 286331153 }

// lldb-command:print case3
// lldbg-check:[...]$2 = Case3 { a: 0, b: 6438275382588823897 }
// lldbr-check:(struct_style_enum_legacy::Regular::Case3) case3 = Case3 { struct_style_enum_legacy::Regular::Case1: 0, struct_style_enum_legacy::Regular::Case2: 6438275382588823897 }

// lldb-command:print univariant
// lldbg-check:[...]$3 = TheOnlyCase { a: -1 }
// lldbr-check:(struct_style_enum_legacy::Univariant) univariant = Univariant { struct_style_enum_legacy::TheOnlyCase: TheOnlyCase { a: -1 } }

#![allow(unused_variables)]
#![feature(omit_gdb_pretty_printer_section)]
#![omit_gdb_pretty_printer_section]

use self::Regular::{Case1, Case2, Case3};
use self::Univariant::TheOnlyCase;

// The first element is to ensure proper alignment, irrespective of the machines word size. Since
// the size of the discriminant value is machine dependent, this has be taken into account when
// datatype layout should be predictable as in this case.
enum Regular {
    Case1 { a: u64, b: u16, c: u16, d: u16, e: u16},
    Case2 { a: u64, b: u32, c: u32},
    Case3 { a: u64, b: u64 }
}

enum Univariant {
    TheOnlyCase { a: i64 }
}

fn main() {

    // In order to avoid endianness trouble all of the following test values consist of a single
    // repeated byte. This way each interpretation of the union should look the same, no matter if
    // this is a big or little endian machine.

    // 0b0111110001111100011111000111110001111100011111000111110001111100 = 8970181431921507452
    // 0b01111100011111000111110001111100 = 2088533116
    // 0b0111110001111100 = 31868
    // 0b01111100 = 124
    let case1 = Case1 { a: 0, b: 31868, c: 31868, d: 31868, e: 31868 };

    // 0b0001000100010001000100010001000100010001000100010001000100010001 = 1229782938247303441
    // 0b00010001000100010001000100010001 = 286331153
    // 0b0001000100010001 = 4369
    // 0b00010001 = 17
    let case2 = Case2 { a: 0, b: 286331153, c: 286331153 };

    // 0b0101100101011001010110010101100101011001010110010101100101011001 = 6438275382588823897
    // 0b01011001010110010101100101011001 = 1499027801
    // 0b0101100101011001 = 22873
    // 0b01011001 = 89
    let case3 = Case3 { a: 0, b: 6438275382588823897 };

    let univariant = TheOnlyCase { a: -1 };

    zzz(); // #break
}

fn zzz() {()}
