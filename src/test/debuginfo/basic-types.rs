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

// min-lldb-version: 310

// compile-flags:-g

// === GDB TESTS ===================================================================================

// gdb-command:run
// gdb-command:print b
// gdb-check:$1 = false
// gdb-command:print i
// gdb-check:$2 = -1
// gdb-command:print c
// gdbg-check:$3 = 97
// gdbr-check:$3 = 97 'a'
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


// === LLDB TESTS ==================================================================================

// lldb-command:run
// lldb-command:print b
// lldb-check:[...]$0 = false
// lldb-command:print i
// lldb-check:[...]$1 = -1

// NOTE: LLDB does not support 32bit chars
// d ebugger:print (usize)(c)
// c heck:$3 = 97

// lldb-command:print i8
// lldb-check:[...]$2 = 'D'
// lldb-command:print i16
// lldb-check:[...]$3 = -16
// lldb-command:print i32
// lldb-check:[...]$4 = -32
// lldb-command:print i64
// lldb-check:[...]$5 = -64
// lldb-command:print u
// lldb-check:[...]$6 = 1
// lldb-command:print u8
// lldb-check:[...]$7 = 'd'
// lldb-command:print u16
// lldb-check:[...]$8 = 16
// lldb-command:print u32
// lldb-check:[...]$9 = 32
// lldb-command:print u64
// lldb-check:[...]$10 = 64
// lldb-command:print f32
// lldb-check:[...]$11 = 2.5
// lldb-command:print f64
// lldb-check:[...]$12 = 3.5

#![allow(unused_variables)]
#![feature(omit_gdb_pretty_printer_section)]
#![omit_gdb_pretty_printer_section]

fn main() {
    let b: bool = false;
    let i: isize = -1;
    let c: char = 'a';
    let i8: i8 = 68;
    let i16: i16 = -16;
    let i32: i32 = -32;
    let i64: i64 = -64;
    let u: usize = 1;
    let u8: u8 = 100;
    let u16: u16 = 16;
    let u32: u32 = 32;
    let u64: u64 = 64;
    let f32: f32 = 2.5;
    let f64: f64 = 3.5;
    _zzz(); // #break
}

fn _zzz() {()}
