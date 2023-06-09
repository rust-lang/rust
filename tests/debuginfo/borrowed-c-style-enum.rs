// compile-flags:-g
// min-lldb-version: 310

// === GDB TESTS ===================================================================================

// gdb-command:run

// gdb-command:print *the_a_ref
// gdbg-check:$1 = TheA
// gdbr-check:$1 = borrowed_c_style_enum::ABC::TheA

// gdb-command:print *the_b_ref
// gdbg-check:$2 = TheB
// gdbr-check:$2 = borrowed_c_style_enum::ABC::TheB

// gdb-command:print *the_c_ref
// gdbg-check:$3 = TheC
// gdbr-check:$3 = borrowed_c_style_enum::ABC::TheC


// === LLDB TESTS ==================================================================================

// lldb-command:run

// lldb-command:print *the_a_ref
// lldbg-check:[...]$0 = TheA
// lldbr-check:(borrowed_c_style_enum::ABC) *the_a_ref = borrowed_c_style_enum::ABC::TheA

// lldb-command:print *the_b_ref
// lldbg-check:[...]$1 = TheB
// lldbr-check:(borrowed_c_style_enum::ABC) *the_b_ref = borrowed_c_style_enum::ABC::TheB

// lldb-command:print *the_c_ref
// lldbg-check:[...]$2 = TheC
// lldbr-check:(borrowed_c_style_enum::ABC) *the_c_ref = borrowed_c_style_enum::ABC::TheC

#![allow(unused_variables)]
#![feature(omit_gdb_pretty_printer_section)]
#![omit_gdb_pretty_printer_section]

enum ABC { TheA, TheB, TheC }

fn main() {
    let the_a = ABC::TheA;
    let the_a_ref: &ABC = &the_a;

    let the_b = ABC::TheB;
    let the_b_ref: &ABC = &the_b;

    let the_c = ABC::TheC;
    let the_c_ref: &ABC = &the_c;

    zzz(); // #break
}

fn zzz() {()}
