// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-win32: FIXME #13256
// ignore-android: FIXME(#10381)

// compile-flags:-g
// gdb-command:rbreak zzz
// gdb-command:run
// gdb-command:finish
// gdb-command:whatis 'basic-types-globals-metadata::B'
// gdb-check:type = bool
// gdb-command:whatis 'basic-types-globals-metadata::I'
// gdb-check:type = int
// gdb-command:whatis 'basic-types-globals-metadata::C'
// gdb-check:type = char
// gdb-command:whatis 'basic-types-globals-metadata::I8'
// gdb-check:type = i8
// gdb-command:whatis 'basic-types-globals-metadata::I16'
// gdb-check:type = i16
// gdb-command:whatis 'basic-types-globals-metadata::I32'
// gdb-check:type = i32
// gdb-command:whatis 'basic-types-globals-metadata::I64'
// gdb-check:type = i64
// gdb-command:whatis 'basic-types-globals-metadata::U'
// gdb-check:type = uint
// gdb-command:whatis 'basic-types-globals-metadata::U8'
// gdb-check:type = u8
// gdb-command:whatis 'basic-types-globals-metadata::U16'
// gdb-check:type = u16
// gdb-command:whatis 'basic-types-globals-metadata::U32'
// gdb-check:type = u32
// gdb-command:whatis 'basic-types-globals-metadata::U64'
// gdb-check:type = u64
// gdb-command:whatis 'basic-types-globals-metadata::F32'
// gdb-check:type = f32
// gdb-command:whatis 'basic-types-globals-metadata::F64'
// gdb-check:type = f64
// gdb-command:continue

#![allow(unused_variable)]
#![allow(dead_code)]


static B: bool = false;
static I: int = -1;
static C: char = 'a';
static I8: i8 = 68;
static I16: i16 = -16;
static I32: i32 = -32;
static I64: i64 = -64;
static U: uint = 1;
static U8: u8 = 100;
static U16: u16 = 16;
static U32: u32 = 32;
static U64: u64 = 64;
static F32: f32 = 2.5;
static F64: f64 = 3.5;

fn main() {
    _zzz();

    let a = (B, I, C, I8, I16, I32, I64, U, U8, U16, U32, U64, F32, F64);
}

fn _zzz() {()}
