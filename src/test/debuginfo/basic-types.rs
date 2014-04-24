// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Caveats - gdb prints any 8-bit value (meaning rust i8 and u8 values)
// as its numerical value along with its associated ASCII char, there
// doesn't seem to be any way around this. Also, gdb doesn't know
// about UTF-32 character encoding and will print a rust char as only
// its numerical value.

// ignore-android: FIXME(#10381)

// compile-flags:-g
// gdb-command:rbreak zzz
// gdb-command:run
// gdb-command:finish
// gdb-command:print b
// gdb-check:$1 = false
// gdb-command:print i
// gdb-check:$2 = -1
// gdb-command:print c
// gdb-check:$3 = 97
// gdb-command:print/d i8
// gdb-check:$4 = 68
// gdb-command:print i16
// gdb-check:$5 = -16
// gdb-command:print i32
// gdb-check:$6 = -32
// gdb-command:print i64
// gdb-check:$7 = -64
// gdb-command:print u
// gdb-check:$8 = 1
// gdb-command:print/d u8
// gdb-check:$9 = 100
// gdb-command:print u16
// gdb-check:$10 = 16
// gdb-command:print u32
// gdb-check:$11 = 32
// gdb-command:print u64
// gdb-check:$12 = 64
// gdb-command:print f32
// gdb-check:$13 = 2.5
// gdb-command:print f64
// gdb-check:$14 = 3.5

#![allow(unused_variable)]

fn main() {
    let b: bool = false;
    let i: int = -1;
    let c: char = 'a';
    let i8: i8 = 68;
    let i16: i16 = -16;
    let i32: i32 = -32;
    let i64: i64 = -64;
    let u: uint = 1;
    let u8: u8 = 100;
    let u16: u16 = 16;
    let u32: u32 = 32;
    let u64: u64 = 64;
    let f32: f32 = 2.5;
    let f64: f64 = 3.5;
    _zzz();
}

fn _zzz() {()}
