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

// === GDB TESTS ===================================================================================

// gdb-command:run
// gdb-command:print a
// gdbg-check:$1 = {1, 2, 3}
// gdbr-check:$1 = [1, 2, 3]
// gdb-command:print vec::VECT
// gdbg-check:$2 = {4, 5, 6}
// gdbr-check:$2 = [4, 5, 6]


// === LLDB TESTS ==================================================================================

// lldb-command:run
// lldb-command:print a
// lldb-check:[...]$0 = [1, 2, 3]

#![allow(unused_variables)]
#![feature(omit_gdb_pretty_printer_section)]
#![omit_gdb_pretty_printer_section]

static mut VECT: [i32; 3] = [1, 2, 3];

fn main() {
    let a = [1, 2, 3];

    unsafe {
        VECT[0] = 4;
        VECT[1] = 5;
        VECT[2] = 6;
    }

    zzz(); // #break
}

fn zzz() {()}
