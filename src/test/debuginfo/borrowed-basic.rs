// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-android: FIXME(#10381)

// Gdb doesn't know about UTF-32 character encoding and will print a rust char as only
// its numerical value.

// compile-flags:-g

// === GDB TESTS ===================================================================================

// gdb-command:rbreak zzz
// gdb-command:run
// gdb-command:finish
// gdb-command:print *bool_ref
// gdb-check:$1 = true

// gdb-command:print *int_ref
// gdb-check:$2 = -1

// gdb-command:print *char_ref
// gdb-check:$3 = 97

// gdb-command:print *i8_ref
// gdb-check:$4 = 68 'D'

// gdb-command:print *i16_ref
// gdb-check:$5 = -16

// gdb-command:print *i32_ref
// gdb-check:$6 = -32

// gdb-command:print *i64_ref
// gdb-check:$7 = -64

// gdb-command:print *uint_ref
// gdb-check:$8 = 1

// gdb-command:print *u8_ref
// gdb-check:$9 = 100 'd'

// gdb-command:print *u16_ref
// gdb-check:$10 = 16

// gdb-command:print *u32_ref
// gdb-check:$11 = 32

// gdb-command:print *u64_ref
// gdb-check:$12 = 64

// gdb-command:print *f32_ref
// gdb-check:$13 = 2.5

// gdb-command:print *f64_ref
// gdb-check:$14 = 3.5


// === LLDB TESTS ==================================================================================

// lldb-command:run
// lldb-command:print *bool_ref
// lldb-check:[...]$0 = true

// lldb-command:print *int_ref
// lldb-check:[...]$1 = -1

// NOTE: lldb doesn't support 32bit chars at the moment
// d ebugger:print *char_ref
// c heck:[...]$x = 97

// lldb-command:print *i8_ref
// lldb-check:[...]$2 = 'D'

// lldb-command:print *i16_ref
// lldb-check:[...]$3 = -16

// lldb-command:print *i32_ref
// lldb-check:[...]$4 = -32

// lldb-command:print *i64_ref
// lldb-check:[...]$5 = -64

// lldb-command:print *uint_ref
// lldb-check:[...]$6 = 1

// lldb-command:print *u8_ref
// lldb-check:[...]$7 = 'd'

// lldb-command:print *u16_ref
// lldb-check:[...]$8 = 16

// lldb-command:print *u32_ref
// lldb-check:[...]$9 = 32

// lldb-command:print *u64_ref
// lldb-check:[...]$10 = 64

// lldb-command:print *f32_ref
// lldb-check:[...]$11 = 2.5

// lldb-command:print *f64_ref
// lldb-check:[...]$12 = 3.5

#![allow(unused_variable)]

fn main() {
    let bool_val: bool = true;
    let bool_ref: &bool = &bool_val;

    let int_val: int = -1;
    let int_ref: &int = &int_val;

    let char_val: char = 'a';
    let char_ref: &char = &char_val;

    let i8_val: i8 = 68;
    let i8_ref: &i8 = &i8_val;

    let i16_val: i16 = -16;
    let i16_ref: &i16 = &i16_val;

    let i32_val: i32 = -32;
    let i32_ref: &i32 = &i32_val;

    let uint_val: i64 = -64;
    let i64_ref: &i64 = &uint_val;

    let uint_val: uint = 1;
    let uint_ref: &uint = &uint_val;

    let u8_val: u8 = 100;
    let u8_ref: &u8 = &u8_val;

    let u16_val: u16 = 16;
    let u16_ref: &u16 = &u16_val;

    let u32_val: u32 = 32;
    let u32_ref: &u32 = &u32_val;

    let u64_val: u64 = 64;
    let u64_ref: &u64 = &u64_val;

    let f32_val: f32 = 2.5;
    let f32_ref: &f32 = &f32_val;

    let f64_val: f64 = 3.5;
    let f64_ref: &f64 = &f64_val;

    zzz(); // #break
}

fn zzz() {()}
