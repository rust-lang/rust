//@ compile-flags:-g
//@ disable-gdb-pretty-printers

// === GDB TESTS ===================================================================================

// gdb-command:run

// gdb-command:print *the_a_ref
// gdb-check:$1 = borrowed_c_style_enum::ABC::TheA

// gdb-command:print *the_b_ref
// gdb-check:$2 = borrowed_c_style_enum::ABC::TheB

// gdb-command:print *the_c_ref
// gdb-check:$3 = borrowed_c_style_enum::ABC::TheC


// === LLDB TESTS ==================================================================================

// lldb-command:run

// lldb-command:v *the_a_ref
// lldb-check:[...] TheA

// lldb-command:v *the_b_ref
// lldb-check:[...] TheB

// lldb-command:v *the_c_ref
// lldb-check:[...] TheC

#![allow(unused_variables)]

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
