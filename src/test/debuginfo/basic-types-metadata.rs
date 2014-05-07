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

// compile-flags:-g
// gdb-command:rbreak zzz
// gdb-command:run
// gdb-command:finish
// gdb-command:whatis unit
// gdb-check:type = ()
// gdb-command:whatis b
// gdb-check:type = bool
// gdb-command:whatis i
// gdb-check:type = int
// gdb-command:whatis c
// gdb-check:type = char
// gdb-command:whatis i8
// gdb-check:type = i8
// gdb-command:whatis i16
// gdb-check:type = i16
// gdb-command:whatis i32
// gdb-check:type = i32
// gdb-command:whatis i64
// gdb-check:type = i64
// gdb-command:whatis u
// gdb-check:type = uint
// gdb-command:whatis u8
// gdb-check:type = u8
// gdb-command:whatis u16
// gdb-check:type = u16
// gdb-command:whatis u32
// gdb-check:type = u32
// gdb-command:whatis u64
// gdb-check:type = u64
// gdb-command:whatis f32
// gdb-check:type = f32
// gdb-command:whatis f64
// gdb-check:type = f64
// gdb-command:info functions _yyy
// gdb-check:[...]![...]_yyy([...])([...]);
// gdb-command:continue

#![allow(unused_variable)]

fn main() {
    let unit: () = ();
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
    if 1 == 1 { _yyy(); }
}

fn _zzz() {()}
fn _yyy() -> ! {fail!()}
