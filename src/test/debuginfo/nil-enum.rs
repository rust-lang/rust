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

// gdb-command:print *first
// gdbg-check:$1 = {<No data fields>}
// gdbr-check:$1 = <error reading variable>

#![allow(unused_variables)]
#![feature(omit_gdb_pretty_printer_section)]
#![omit_gdb_pretty_printer_section]

enum Void {}

fn main() {
    let first: *const Void = 1 as *const _;

    zzz(); // #break
}

fn zzz() {}
