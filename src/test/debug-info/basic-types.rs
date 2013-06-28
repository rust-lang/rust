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
// debugger:break _zzz
// debugger:run
// debugger:finish
// debugger:print b
// check:$1 = false
// debugger:print i
// check:$2 = -1
// debugger:print c
// check:$3 = 97
// debugger:print/d i8
// check:$4 = 68
// debugger:print i16
// check:$5 = -16
// debugger:print i32
// check:$6 = -32
// debugger:print i64
// check:$7 = -64
// debugger:print u
// check:$8 = 1
// debugger:print/d u8
// check:$9 = 100
// debugger:print u16
// check:$10 = 16
// debugger:print u32
// check:$11 = 32
// debugger:print u64
// check:$12 = 64
// debugger:print f
// check:$13 = 1.5
// debugger:print f32
// check:$14 = 2.5
// debugger:print f64
// check:$15 = 3.5

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
    let f: float = 1.5;
    let f32: f32 = 2.5;
    let f64: f64 = 3.5;
    _zzz();
}

fn _zzz() {()}
