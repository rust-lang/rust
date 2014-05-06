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
// debugger:rbreak zzz
// debugger:run
// debugger:finish
// debugger:print *bool_ref
// check:$1 = true

// debugger:print *int_ref
// check:$2 = -1

// debugger:print *char_ref
// check:$3 = 97

// debugger:print/d *i8_ref
// check:$4 = 68

// debugger:print *i16_ref
// check:$5 = -16

// debugger:print *i32_ref
// check:$6 = -32

// debugger:print *i64_ref
// check:$7 = -64

// debugger:print *uint_ref
// check:$8 = 1

// debugger:print/d *u8_ref
// check:$9 = 100

// debugger:print *u16_ref
// check:$10 = 16

// debugger:print *u32_ref
// check:$11 = 32

// debugger:print *u64_ref
// check:$12 = 64

// debugger:print *f32_ref
// check:$13 = 2.5

// debugger:print *f64_ref
// check:$14 = 3.5

#![allow(unused_variable)]


fn main() {
    let bool_box: Box<bool> = box true;
    let bool_ref: &bool = bool_box;

    let int_box: Box<int> = box -1;
    let int_ref: &int = int_box;

    let char_box: Box<char> = box 'a';
    let char_ref: &char = char_box;

    let i8_box: Box<i8> = box 68;
    let i8_ref: &i8 = i8_box;

    let i16_box: Box<i16> = box -16;
    let i16_ref: &i16 = i16_box;

    let i32_box: Box<i32> = box -32;
    let i32_ref: &i32 = i32_box;

    let i64_box: Box<i64> = box -64;
    let i64_ref: &i64 = i64_box;

    let uint_box: Box<uint> = box 1;
    let uint_ref: &uint = uint_box;

    let u8_box: Box<u8> = box 100;
    let u8_ref: &u8 = u8_box;

    let u16_box: Box<u16> = box 16;
    let u16_ref: &u16 = u16_box;

    let u32_box: Box<u32> = box 32;
    let u32_ref: &u32 = u32_box;

    let u64_box: Box<u64> = box 64;
    let u64_ref: &u64 = u64_box;

    let f32_box: Box<f32> = box 2.5;
    let f32_ref: &f32 = f32_box;

    let f64_box: Box<f64> = box 3.5;
    let f64_ref: &f64 = f64_box;
    zzz();
}

fn zzz() {()}
