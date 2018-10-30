// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// NOTE Instantiating an empty enum is UB. This test may break in the future.

// LLDB can't handle zero-sized values
// ignore-lldb


// Require LLVM with DW_TAG_variant_part and a gdb that can read it.
// gdb 8.2.0 crashes on this test case, see
// https://sourceware.org/bugzilla/show_bug.cgi?id=23626
// This will be fixed in the next release, which will be >= 8.2.1.
// min-system-llvm-version: 7.0
// min-gdb-version: 8.2.1

// compile-flags:-g
// gdb-command:run

// gdb-command:print first
// gdbr-check:$1 = nil_enum::ANilEnum {<No data fields>}

// gdb-command:print second
// gdbr-check:$2 = nil_enum::AnotherNilEnum {<No data fields>}

#![allow(unused_variables)]
#![feature(omit_gdb_pretty_printer_section)]
#![feature(maybe_uninit)]
#![omit_gdb_pretty_printer_section]

use std::mem::MaybeUninit;

enum ANilEnum {}
enum AnotherNilEnum {}

// This test relies on gdbg printing the string "{<No data fields>}" for empty
// structs (which may change some time)
// The error from gdbr is expected since nil enums are not supposed to exist.
fn main() {
    unsafe {
        let first: ANilEnum = MaybeUninit::uninitialized().into_inner();
        let second: AnotherNilEnum = MaybeUninit::uninitialized().into_inner();

        zzz(); // #break
    }
}

fn zzz() {()}
