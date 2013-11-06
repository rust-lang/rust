// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-android

// compile-flags:-Z extra-debug-info
// debugger:rbreak zzz
// debugger:run
// debugger:finish
// debugger:whatis unit
// check:type = ()
// debugger:whatis b
// check:type = bool
// debugger:whatis i
// check:type = int
// debugger:whatis c
// check:type = char
// debugger:whatis i8
// check:type = i8
// debugger:whatis i16
// check:type = i16
// debugger:whatis i32
// check:type = i32
// debugger:whatis i64
// check:type = i64
// debugger:whatis u
// check:type = uint
// debugger:whatis u8
// check:type = u8
// debugger:whatis u16
// check:type = u16
// debugger:whatis u32
// check:type = u32
// debugger:whatis u64
// check:type = u64
// debugger:whatis f32
// check:type = f32
// debugger:whatis f64
// check:type = f64
// debugger:info functions _yyy
// check:[...]
// check:![...]_yyy()();
// debugger:detach
// debugger:quit

#[allow(unused_variable)];

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
}

fn _zzz() {()}
fn _yyy() -> ! {fail!()}
