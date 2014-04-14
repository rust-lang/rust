// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Caveats - gdb prints any 8-bit value (meaning rust I8 and u8 values)
// as its numerical value along with its associated ASCII char, there
// doesn't seem to be any way around this. Also, gdb doesn't know
// about UTF-32 character encoding and will print a rust char as only
// its numerical value.

// ignore-win32: FIXME #13256
// ignore-android: FIXME(#10381)

// compile-flags:-g
// debugger:rbreak zzz
// debugger:run
// debugger:finish

// Check initializers
// debugger:print 'basic-types-mut-globals::B'
// check:$1 = false
// debugger:print 'basic-types-mut-globals::I'
// check:$2 = -1
// debugger:print 'basic-types-mut-globals::C'
// check:$3 = 97
// debugger:print/d 'basic-types-mut-globals::I8'
// check:$4 = 68
// debugger:print 'basic-types-mut-globals::I16'
// check:$5 = -16
// debugger:print 'basic-types-mut-globals::I32'
// check:$6 = -32
// debugger:print 'basic-types-mut-globals::I64'
// check:$7 = -64
// debugger:print 'basic-types-mut-globals::U'
// check:$8 = 1
// debugger:print/d 'basic-types-mut-globals::U8'
// check:$9 = 100
// debugger:print 'basic-types-mut-globals::U16'
// check:$10 = 16
// debugger:print 'basic-types-mut-globals::U32'
// check:$11 = 32
// debugger:print 'basic-types-mut-globals::U64'
// check:$12 = 64
// debugger:print 'basic-types-mut-globals::F32'
// check:$13 = 2.5
// debugger:print 'basic-types-mut-globals::F64'
// check:$14 = 3.5
// debugger:continue

// Check new values
// debugger:print 'basic-types-mut-globals'::B
// check:$15 = true
// debugger:print 'basic-types-mut-globals'::I
// check:$16 = 2
// debugger:print 'basic-types-mut-globals'::C
// check:$17 = 102
// debugger:print/d 'basic-types-mut-globals'::I8
// check:$18 = 78
// debugger:print 'basic-types-mut-globals'::I16
// check:$19 = -26
// debugger:print 'basic-types-mut-globals'::I32
// check:$20 = -12
// debugger:print 'basic-types-mut-globals'::I64
// check:$21 = -54
// debugger:print 'basic-types-mut-globals'::U
// check:$22 = 5
// debugger:print/d 'basic-types-mut-globals'::U8
// check:$23 = 20
// debugger:print 'basic-types-mut-globals'::U16
// check:$24 = 32
// debugger:print 'basic-types-mut-globals'::U32
// check:$25 = 16
// debugger:print 'basic-types-mut-globals'::U64
// check:$26 = 128
// debugger:print 'basic-types-mut-globals'::F32
// check:$27 = 5.75
// debugger:print 'basic-types-mut-globals'::F64
// check:$28 = 9.25

// debugger:detach
// debugger:quit

#![allow(unused_variable)]

static mut B: bool = false;
static mut I: int = -1;
static mut C: char = 'a';
static mut I8: i8 = 68;
static mut I16: i16 = -16;
static mut I32: i32 = -32;
static mut I64: i64 = -64;
static mut U: uint = 1;
static mut U8: u8 = 100;
static mut U16: u16 = 16;
static mut U32: u32 = 32;
static mut U64: u64 = 64;
static mut F32: f32 = 2.5;
static mut F64: f64 = 3.5;

fn main() {
    _zzz();

    unsafe {
        B = true;
        I = 2;
        C = 'f';
        I8 = 78;
        I16 = -26;
        I32 = -12;
        I64 = -54;
        U = 5;
        U8 = 20;
        U16 = 32;
        U32 = 16;
        U64 = 128;
        F32 = 5.75;
        F64 = 9.25;
    }

    _zzz();
}

fn _zzz() {()}
