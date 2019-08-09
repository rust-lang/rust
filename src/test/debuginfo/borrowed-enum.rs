// ignore-tidy-linelength

// Require LLVM with DW_TAG_variant_part and a gdb or lldb that can read it.
// min-system-llvm-version: 8.0
// min-gdb-version: 8.2
// rust-lldb

// compile-flags:-g

// === GDB TESTS ===================================================================================

// gdb-command:run

// gdb-command:print *the_a_ref
// gdbr-check:$1 = borrowed_enum::ABC::TheA{x: 0, y: 8970181431921507452}

// gdb-command:print *the_b_ref
// gdbr-check:$2 = borrowed_enum::ABC::TheB(0, 286331153, 286331153)

// gdb-command:print *univariant_ref
// gdbr-check:$3 = borrowed_enum::Univariant::TheOnlyCase(4820353753753434)


// === LLDB TESTS ==================================================================================

// lldb-command:run

// lldb-command:print *the_a_ref
// lldbr-check:(borrowed_enum::ABC::TheA) *the_a_ref = TheA { TheA: 0, TheB: 8970181431921507452 }
// lldb-command:print *the_b_ref
// lldbr-check:(borrowed_enum::ABC::TheB) *the_b_ref = { = 0 = 286331153 = 286331153 }
// lldb-command:print *univariant_ref
// lldbr-check:(borrowed_enum::Univariant) *univariant_ref = { TheOnlyCase = { = 4820353753753434 } }

#![allow(unused_variables)]
#![feature(omit_gdb_pretty_printer_section)]
#![omit_gdb_pretty_printer_section]

// The first element is to ensure proper alignment, irrespective of the machines word size. Since
// the size of the discriminant value is machine dependent, this has be taken into account when
// datatype layout should be predictable as in this case.
enum ABC {
    TheA { x: i64, y: i64 },
    TheB (i64, i32, i32),
}

// This is a special case since it does not have the implicit discriminant field.
enum Univariant {
    TheOnlyCase(i64)
}

fn main() {

    // 0b0111110001111100011111000111110001111100011111000111110001111100 = 8970181431921507452
    // 0b01111100011111000111110001111100 = 2088533116
    // 0b0111110001111100 = 31868
    // 0b01111100 = 124
    let the_a = ABC::TheA { x: 0, y: 8970181431921507452 };
    let the_a_ref: &ABC = &the_a;

    // 0b0001000100010001000100010001000100010001000100010001000100010001 = 1229782938247303441
    // 0b00010001000100010001000100010001 = 286331153
    // 0b0001000100010001 = 4369
    // 0b00010001 = 17
    let the_b = ABC::TheB (0, 286331153, 286331153);
    let the_b_ref: &ABC = &the_b;

    let univariant = Univariant::TheOnlyCase(4820353753753434);
    let univariant_ref: &Univariant = &univariant;

    zzz(); // #break
}

fn zzz() {()}
