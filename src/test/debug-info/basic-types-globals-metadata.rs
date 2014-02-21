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
// debugger:rbreak zzz
// debugger:run
// debugger:finish
// debugger:whatis 'basic-types-globals-metadata::B'
// check:type = bool
// debugger:whatis 'basic-types-globals-metadata::I'
// check:type = int
// debugger:whatis 'basic-types-globals-metadata::C'
// check:type = char
// debugger:whatis 'basic-types-globals-metadata::I8'
// check:type = i8
// debugger:whatis 'basic-types-globals-metadata::I16'
// check:type = i16
// debugger:whatis 'basic-types-globals-metadata::I32'
// check:type = i32
// debugger:whatis 'basic-types-globals-metadata::I64'
// check:type = i64
// debugger:whatis 'basic-types-globals-metadata::U'
// check:type = uint
// debugger:whatis 'basic-types-globals-metadata::U8'
// check:type = u8
// debugger:whatis 'basic-types-globals-metadata::U16'
// check:type = u16
// debugger:whatis 'basic-types-globals-metadata::U32'
// check:type = u32
// debugger:whatis 'basic-types-globals-metadata::U64'
// check:type = u64
// debugger:whatis 'basic-types-globals-metadata::F32'
// check:type = f32
// debugger:whatis 'basic-types-globals-metadata::F64'
// check:type = f64
// debugger:continue

#[allow(unused_variable)];

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
}

fn _zzz() {()}
