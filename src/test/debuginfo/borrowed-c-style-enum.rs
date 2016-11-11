// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


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
// lldb-check:[...]$0 = TheA

// lldb-command:print *the_b_ref
// lldb-check:[...]$1 = TheB

// lldb-command:print *the_c_ref
// lldb-check:[...]$2 = TheC

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
