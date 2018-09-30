// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// LLDB can't handle zero-sized values
// ignore-lldb


// compile-flags:-g
// gdb-command:run

// gdb-command:print first
// gdbg-check:$1 = {<No data fields>}
// gdbr-check:$1 = <error reading variable>

// gdb-command:print second
// gdbg-check:$2 = {<No data fields>}
// gdbr-check:$2 = <error reading variable>

#![allow(unused_variables)]
#![feature(omit_gdb_pretty_printer_section)]
#![omit_gdb_pretty_printer_section]

enum ANilEnum {}
enum AnotherNilEnum {}

// This test relies on gdbg printing the string "{<No data fields>}" for empty
// structs (which may change some time)
// The error from gdbr is expected since nil enums are not supposed to exist.
fn main() {
    unsafe {
        let first: ANilEnum = ::std::mem::zeroed();
        let second: AnotherNilEnum = ::std::mem::zeroed();

        zzz(); // #break
    }
}

fn zzz() {()}
