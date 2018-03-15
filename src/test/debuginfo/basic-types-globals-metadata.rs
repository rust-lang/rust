// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// min-lldb-version: 310

// compile-flags:-g
// gdb-command:run
// gdbg-command:whatis 'basic_types_globals_metadata::B'
// gdbr-command:whatis basic_types_globals_metadata::B
// gdb-check:type = bool
// gdbg-command:whatis 'basic_types_globals_metadata::I'
// gdbr-command:whatis basic_types_globals_metadata::I
// gdb-check:type = isize
// gdbg-command:whatis 'basic_types_globals_metadata::C'
// gdbr-command:whatis basic_types_globals_metadata::C
// gdb-check:type = char
// gdbg-command:whatis 'basic_types_globals_metadata::I8'
// gdbr-command:whatis basic_types_globals_metadata::I8
// gdb-check:type = i8
// gdbg-command:whatis 'basic_types_globals_metadata::I16'
// gdbr-command:whatis basic_types_globals_metadata::I16
// gdb-check:type = i16
// gdbg-command:whatis 'basic_types_globals_metadata::I32'
// gdbr-command:whatis basic_types_globals_metadata::I32
// gdb-check:type = i32
// gdbg-command:whatis 'basic_types_globals_metadata::I64'
// gdbr-command:whatis basic_types_globals_metadata::I64
// gdb-check:type = i64
// gdbg-command:whatis 'basic_types_globals_metadata::U'
// gdbr-command:whatis basic_types_globals_metadata::U
// gdb-check:type = usize
// gdbg-command:whatis 'basic_types_globals_metadata::U8'
// gdbr-command:whatis basic_types_globals_metadata::U8
// gdb-check:type = u8
// gdbg-command:whatis 'basic_types_globals_metadata::U16'
// gdbr-command:whatis basic_types_globals_metadata::U16
// gdb-check:type = u16
// gdbg-command:whatis 'basic_types_globals_metadata::U32'
// gdbr-command:whatis basic_types_globals_metadata::U32
// gdb-check:type = u32
// gdbg-command:whatis 'basic_types_globals_metadata::U64'
// gdbr-command:whatis basic_types_globals_metadata::U64
// gdb-check:type = u64
// gdbg-command:whatis 'basic_types_globals_metadata::F32'
// gdbr-command:whatis basic_types_globals_metadata::F32
// gdb-check:type = f32
// gdbg-command:whatis 'basic_types_globals_metadata::F64'
// gdbr-command:whatis basic_types_globals_metadata::F64
// gdb-check:type = f64
// gdb-command:continue

#![allow(unused_variables)]
#![allow(dead_code)]
#![feature(omit_gdb_pretty_printer_section)]
#![omit_gdb_pretty_printer_section]

// N.B. These are `mut` only so they don't constant fold away.
static mut B: bool = false;
static mut I: isize = -1;
static mut C: char = 'a';
static mut I8: i8 = 68;
static mut I16: i16 = -16;
static mut I32: i32 = -32;
static mut I64: i64 = -64;
static mut U: usize = 1;
static mut U8: u8 = 100;
static mut U16: u16 = 16;
static mut U32: u32 = 32;
static mut U64: u64 = 64;
static mut F32: f32 = 2.5;
static mut F64: f64 = 3.5;

fn main() {
    _zzz(); // #break

    let a = unsafe { (B, I, C, I8, I16, I32, I64, U, U8, U16, U32, U64, F32, F64) };
}

fn _zzz() {()}
