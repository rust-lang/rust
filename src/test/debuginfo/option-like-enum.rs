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

// compile-flags:-g

// === GDB TESTS ===================================================================================

// gdb-command:run

// gdb-command:print some
// gdbg-check:$1 = {RUST$ENCODED$ENUM$0$None = {__0 = 0x12345678}}
// gdbr-check:$1 = core::option::Option<&u32>::Some(0x12345678)

// gdb-command:print none
// gdbg-check:$2 = {RUST$ENCODED$ENUM$0$None = {__0 = 0x0}}
// gdbr-check:$2 = core::option::Option<&u32>::None

// gdb-command:print full
// gdbg-check:$3 = {RUST$ENCODED$ENUM$1$Empty = {__0 = 454545, __1 = 0x87654321, __2 = 9988}}
// gdbr-check:$3 = option_like_enum::MoreFields::Full(454545, 0x87654321, 9988)

// gdbg-command:print empty_gdb->discr
// gdbr-command:print empty_gdb.discr
// gdb-check:$4 = (isize *) 0x0

// gdb-command:print droid
// gdbg-check:$5 = {RUST$ENCODED$ENUM$2$Void = {id = 675675, range = 10000001, internals = 0x43218765}}
// gdbr-check:$5 = option_like_enum::NamedFields::Droid{id: 675675, range: 10000001, internals: 0x43218765}

// gdbg-command:print void_droid_gdb->internals
// gdbr-command:print void_droid_gdb.internals
// gdb-check:$6 = (isize *) 0x0

// gdb-command:print nested_non_zero_yep
// gdbg-check:$7 = {RUST$ENCODED$ENUM$1$2$Nope = {__0 = 10.5, __1 = {a = 10, b = 20, c = [...]}}}
// gdbr-check:$7 = option_like_enum::NestedNonZero::Yep(10.5, option_like_enum::NestedNonZeroField {a: 10, b: 20, c: 0x[...] "x[...]"})

// gdb-command:print nested_non_zero_nope
// gdbg-check:$8 = {RUST$ENCODED$ENUM$1$2$Nope = {__0 = [...], __1 = {a = [...], b = [...], c = 0x0}}}
// gdbr-check:$8 = option_like_enum::NestedNonZero::Nope

// gdb-command:continue


// === LLDB TESTS ==================================================================================

// lldb-command:run

// lldb-command:print some
// lldb-check:[...]$0 = Some(&0x12345678)

// lldb-command:print none
// lldb-check:[...]$1 = None

// lldb-command:print full
// lldb-check:[...]$2 = Full(454545, &0x87654321, 9988)

// lldb-command:print empty
// lldb-check:[...]$3 = Empty

// lldb-command:print droid
// lldb-check:[...]$4 = Droid { id: 675675, range: 10000001, internals: &0x43218765 }

// lldb-command:print void_droid
// lldb-check:[...]$5 = Void

// lldb-command:print some_str
// lldb-check:[...]$6 = Some("abc")

// lldb-command:print none_str
// lldb-check:[...]$7 = None

// lldb-command:print nested_non_zero_yep
// lldb-check:[...]$8 = Yep(10.5, NestedNonZeroField { a: 10, b: 20, c: &[...] })

// lldb-command:print nested_non_zero_nope
// lldb-check:[...]$9 = Nope


#![feature(omit_gdb_pretty_printer_section)]
#![omit_gdb_pretty_printer_section]

// If a struct has exactly two variants, one of them is empty, and the other one
// contains a non-nullable pointer, then this value is used as the discriminator.
// The test cases in this file make sure that something readable is generated for
// this kind of types.
// If the non-empty variant contains a single non-nullable pointer than the whole
// item is represented as just a pointer and not wrapped in a struct.
// Unfortunately (for these test cases) the content of the non-discriminant fields
// in the null-case is not defined. So we just read the discriminator field in
// this case (by casting the value to a memory-equivalent struct).

enum MoreFields<'a> {
    Full(u32, &'a isize, i16),
    Empty
}

struct MoreFieldsRepr<'a> {
    a: u32,
    discr: &'a isize,
    b: i16
}

enum NamedFields<'a> {
    Droid { id: i32, range: i64, internals: &'a isize },
    Void
}

struct NamedFieldsRepr<'a> {
    id: i32,
    range: i64,
    internals: &'a isize
}

struct NestedNonZeroField<'a> {
    a: u16,
    b: u32,
    c: &'a char,
}

enum NestedNonZero<'a> {
    Yep(f64, NestedNonZeroField<'a>),
    Nope
}

fn main() {

    let some_str: Option<&'static str> = Some("abc");
    let none_str: Option<&'static str> = None;

    let some: Option<&u32> = Some(unsafe { std::mem::transmute(0x12345678_usize) });
    let none: Option<&u32> = None;

    let full = MoreFields::Full(454545, unsafe { std::mem::transmute(0x87654321_usize) }, 9988);

    let empty = MoreFields::Empty;
    let empty_gdb: &MoreFieldsRepr = unsafe { std::mem::transmute(&MoreFields::Empty) };

    let droid = NamedFields::Droid {
        id: 675675,
        range: 10000001,
        internals: unsafe { std::mem::transmute(0x43218765_usize) }
    };

    let void_droid = NamedFields::Void;
    let void_droid_gdb: &NamedFieldsRepr = unsafe { std::mem::transmute(&NamedFields::Void) };

    let x = 'x';
    let nested_non_zero_yep = NestedNonZero::Yep(
        10.5,
        NestedNonZeroField {
            a: 10,
            b: 20,
            c: &x
        });

    let nested_non_zero_nope = NestedNonZero::Nope;

    zzz(); // #break
}

fn zzz() {()}
