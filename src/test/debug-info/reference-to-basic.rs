// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-win32 Broken because of LLVM bug: http://llvm.org/bugs/show_bug.cgi?id=16249

// Caveats - gdb prints any 8-bit value (meaning rust i8 and u8 values)
// as its numerical value along with its associated ASCII char, there
// doesn't seem to be any way around this. Also, gdb doesn't know
// about UTF-32 character encoding and will print a rust char as only
// its numerical value.

// compile-flags:-Z extra-debug-info
// debugger:break zzz
// debugger:run
// debugger:finish
// debugger:print *bool_ref
// check:$1 = true

// debugger:print *int_ref
// check:$2 = -1

// debugger:print *char_ref
// check:$3 = 97

// debugger:print *i8_ref
// check:$4 = 68 'D'

// debugger:print *i16_ref
// check:$5 = -16

// debugger:print *i32_ref
// check:$6 = -32

// debugger:print *i64_ref
// check:$7 = -64

// debugger:print *uint_ref
// check:$8 = 1

// debugger:print *u8_ref
// check:$9 = 100 'd'

// debugger:print *u16_ref
// check:$10 = 16

// debugger:print *u32_ref
// check:$11 = 32

// debugger:print *u64_ref
// check:$12 = 64

// debugger:print *float_ref
// check:$13 = 1.5

// debugger:print *f32_ref
// check:$14 = 2.5

// debugger:print *f64_ref
// check:$15 = 3.5

fn main() {
    let bool_val: bool = true;
    let bool_ref : &bool = &bool_val;

    let int_val: int = -1;
    let int_ref : &int = &int_val;

    let char_val: char = 'a';
    let char_ref : &char = &char_val;

    let i8_val: i8 = 68;
    let i8_ref : &i8 = &i8_val;

    let i16_val: i16 = -16;
    let i16_ref : &i16 = &i16_val;

    let i32_val: i32 = -32;
    let i32_ref : &i32 = &i32_val;

    let uint_val: i64 = -64;
    let i64_ref : &i64 = &uint_val;

    let uint_val: uint = 1;
    let uint_ref : &uint = &uint_val;

    let u8_val: u8 = 100;
    let u8_ref : &u8 = &u8_val;

    let u16_val: u16 = 16;
    let u16_ref : &u16 = &u16_val;

    let u32_val: u32 = 32;
    let u32_ref : &u32 = &u32_val;

    let u64_val: u64 = 64;
    let u64_ref : &u64 = &u64_val;

    let float_val: float = 1.5;
    let float_ref : &float = &float_val;

    let f32_val: f32 = 2.5;
    let f32_ref : &f32 = &f32_val;

    let f64_val: f64 = 3.5;
    let f64_ref : &f64 = &f64_val;
    zzz();
}

fn zzz() {()}