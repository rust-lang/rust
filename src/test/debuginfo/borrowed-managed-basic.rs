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

#![feature(managed_boxes)]

// Gdb doesn't know about UTF-32 character encoding and will print a rust char as only
// its numerical value.

// compile-flags:-g
// gdb-command:rbreak zzz
// gdb-command:run
// gdb-command:finish
// gdb-command:print *bool_ref
// gdb-check:$1 = true

// gdb-command:print *int_ref
// gdb-check:$2 = -1

// gdb-command:print *char_ref
// gdb-check:$3 = 97

// gdb-command:print/d *i8_ref
// gdb-check:$4 = 68

// gdb-command:print *i16_ref
// gdb-check:$5 = -16

// gdb-command:print *i32_ref
// gdb-check:$6 = -32

// gdb-command:print *i64_ref
// gdb-check:$7 = -64

// gdb-command:print *uint_ref
// gdb-check:$8 = 1

// gdb-command:print/d *u8_ref
// gdb-check:$9 = 100

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

#![allow(unused_variable)]

fn main() {
    let bool_box: @bool = @true;
    let bool_ref: &bool = bool_box;

    let int_box: @int = @-1;
    let int_ref: &int = int_box;

    let char_box: @char = @'a';
    let char_ref: &char = char_box;

    let i8_box: @i8 = @68;
    let i8_ref: &i8 = i8_box;

    let i16_box: @i16 = @-16;
    let i16_ref: &i16 = i16_box;

    let i32_box: @i32 = @-32;
    let i32_ref: &i32 = i32_box;

    let i64_box: @i64 = @-64;
    let i64_ref: &i64 = i64_box;

    let uint_box: @uint = @1;
    let uint_ref: &uint = uint_box;

    let u8_box: @u8 = @100;
    let u8_ref: &u8 = u8_box;

    let u16_box: @u16 = @16;
    let u16_ref: &u16 = u16_box;

    let u32_box: @u32 = @32;
    let u32_ref: &u32 = u32_box;

    let u64_box: @u64 = @64;
    let u64_ref: &u64 = u64_box;

    let f32_box: @f32 = @2.5;
    let f32_ref: &f32 = f32_box;

    let f64_box: @f64 = @3.5;
    let f64_ref: &f64 = f64_box;
    zzz();
}

fn zzz() {()}
