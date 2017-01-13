// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-tidy-linelength
// min-lldb-version: 310
// ignore-gdb-version: 7.11.90 - 7.12

// compile-flags:-g

// === GDB TESTS ===================================================================================

// gdb-command:run

// gdb-command:print *the_a
// gdbg-check:$1 = {{RUST$ENUM$DISR = TheA, x = 0, y = 8970181431921507452}, {RUST$ENUM$DISR = TheA, [...]}}
// gdbr-check:$1 = unique_enum::ABC::TheA{x: 0, y: 8970181431921507452}

// gdb-command:print *the_b
// gdbg-check:$2 = {{RUST$ENUM$DISR = TheB, [...]}, {RUST$ENUM$DISR = TheB, __0 = 0, __1 = 286331153, __2 = 286331153}}
// gdbr-check:$2 = unique_enum::ABC::TheB(0, 286331153, 286331153)

// gdb-command:print *univariant
// gdbg-check:$3 = {{__0 = 123234}}
// gdbr-check:$3 = unique_enum::Univariant::TheOnlyCase(123234)


// === LLDB TESTS ==================================================================================

// lldb-command:run

// lldb-command:print *the_a
// lldb-check:[...]$0 = TheA { x: 0, y: 8970181431921507452 }

// lldb-command:print *the_b
// lldb-check:[...]$1 = TheB(0, 286331153, 286331153)

// lldb-command:print *univariant
// lldb-check:[...]$2 = TheOnlyCase(123234)

#![allow(unused_variables)]
#![feature(box_syntax)]
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

    // In order to avoid endianness trouble all of the following test values consist of a single
    // repeated byte. This way each interpretation of the union should look the same, no matter if
    // this is a big or little endian machine.

    // 0b0111110001111100011111000111110001111100011111000111110001111100 = 8970181431921507452
    // 0b01111100011111000111110001111100 = 2088533116
    // 0b0111110001111100 = 31868
    // 0b01111100 = 124
    let the_a: Box<_> = box ABC::TheA { x: 0, y: 8970181431921507452 };

    // 0b0001000100010001000100010001000100010001000100010001000100010001 = 1229782938247303441
    // 0b00010001000100010001000100010001 = 286331153
    // 0b0001000100010001 = 4369
    // 0b00010001 = 17
    let the_b: Box<_> = box ABC::TheB (0, 286331153, 286331153);

    let univariant: Box<_> = box Univariant::TheOnlyCase(123234);

    zzz(); // #break
}

fn zzz() {()}
