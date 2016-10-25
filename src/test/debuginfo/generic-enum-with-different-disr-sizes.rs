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
// ignore-lldb: FIXME(#27089)
// min-lldb-version: 310

// compile-flags:-g

// === GDB TESTS ===================================================================================
// gdb-command:run

// gdb-command:print eight_bytes1
// gdbg-check:$1 = {{RUST$ENUM$DISR = Variant1, __0 = 100}, {RUST$ENUM$DISR = Variant1, __0 = 100}}
// gdbr-check:$1 = generic_enum_with_different_disr_sizes::Enum::Variant1(100)

// gdb-command:print four_bytes1
// gdbg-check:$2 = {{RUST$ENUM$DISR = Variant1, __0 = 101}, {RUST$ENUM$DISR = Variant1, __0 = 101}}
// gdbr-check:$2 = generic_enum_with_different_disr_sizes::Enum::Variant1(101)

// gdb-command:print two_bytes1
// gdbg-check:$3 = {{RUST$ENUM$DISR = Variant1, __0 = 102}, {RUST$ENUM$DISR = Variant1, __0 = 102}}
// gdbr-check:$3 = generic_enum_with_different_disr_sizes::Enum::Variant1(102)

// gdb-command:print one_byte1
// gdbg-check:$4 = {{RUST$ENUM$DISR = Variant1, __0 = 65 'A'}, {RUST$ENUM$DISR = Variant1, __0 = 65 'A'}}
// gdbr-check:$4 = generic_enum_with_different_disr_sizes::Enum::Variant1(65)


// gdb-command:print eight_bytes2
// gdbg-check:$5 = {{RUST$ENUM$DISR = Variant2, __0 = 100}, {RUST$ENUM$DISR = Variant2, __0 = 100}}
// gdbr-check:$5 = generic_enum_with_different_disr_sizes::Enum::Variant2(100)

// gdb-command:print four_bytes2
// gdbg-check:$6 = {{RUST$ENUM$DISR = Variant2, __0 = 101}, {RUST$ENUM$DISR = Variant2, __0 = 101}}
// gdbr-check:$6 = generic_enum_with_different_disr_sizes::Enum::Variant2(101)

// gdb-command:print two_bytes2
// gdbg-check:$7 = {{RUST$ENUM$DISR = Variant2, __0 = 102}, {RUST$ENUM$DISR = Variant2, __0 = 102}}
// gdbr-check:$7 = generic_enum_with_different_disr_sizes::Enum::Variant2(102)

// gdb-command:print one_byte2
// gdbg-check:$8 = {{RUST$ENUM$DISR = Variant2, __0 = 65 'A'}, {RUST$ENUM$DISR = Variant2, __0 = 65 'A'}}
// gdbr-check:$8 = generic_enum_with_different_disr_sizes::Enum::Variant2(65)

// gdb-command:continue

// === LLDB TESTS ==================================================================================
// lldb-command:run

// lldb-command:print eight_bytes1
// lldb-check:[...]$0 = Variant1(100)
// lldb-command:print four_bytes1
// lldb-check:[...]$1 = Variant1(101)
// lldb-command:print two_bytes1
// lldb-check:[...]$2 = Variant1(102)
// lldb-command:print one_byte1
// lldb-check:[...]$3 = Variant1('A')

// lldb-command:print eight_bytes2
// lldb-check:[...]$4 = Variant2(100)
// lldb-command:print four_bytes2
// lldb-check:[...]$5 = Variant2(101)
// lldb-command:print two_bytes2
// lldb-check:[...]$6 = Variant2(102)
// lldb-command:print one_byte2
// lldb-check:[...]$7 = Variant2('A')

// lldb-command:continue

#![allow(unused_variables)]
#![allow(dead_code)]
#![feature(omit_gdb_pretty_printer_section)]
#![omit_gdb_pretty_printer_section]

// This test case makes sure that we get correct type descriptions for the enum
// discriminant of different instantiations of the same generic enum type where,
// dependending on the generic type parameter(s), the discriminant has a
// different size in memory.

enum Enum<T> {
    Variant1(T),
    Variant2(T)
}

fn main() {
    // These are ordered for descending size on purpose
    let eight_bytes1 = Enum::Variant1(100.0f64);
    let four_bytes1 = Enum::Variant1(101i32);
    let two_bytes1 = Enum::Variant1(102i16);
    let one_byte1 = Enum::Variant1(65u8);

    let eight_bytes2 = Enum::Variant2(100.0f64);
    let four_bytes2 = Enum::Variant2(101i32);
    let two_bytes2 = Enum::Variant2(102i16);
    let one_byte2 = Enum::Variant2(65u8);

    zzz(); // #break
}

fn zzz() { () }
