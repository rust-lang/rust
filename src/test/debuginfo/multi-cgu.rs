// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


// This test case makes sure that we get proper break points for binaries
// compiled with multiple codegen units. (see #39160)


// min-lldb-version: 310

// compile-flags:-g -Ccodegen-units=2

// === GDB TESTS ===============================================================

// gdb-command:run

// gdb-command:print xxx
// gdb-check:$1 = 12345
// gdb-command:continue

// gdb-command:print yyy
// gdb-check:$2 = 67890
// gdb-command:continue


// === LLDB TESTS ==============================================================

// lldb-command:run

// lldb-command:print xxx
// lldb-check:[...]$0 = 12345
// lldb-command:continue

// lldb-command:print yyy
// lldb-check:[...]$1 = 67890
// lldb-command:continue


#![feature(omit_gdb_pretty_printer_section)]
#![omit_gdb_pretty_printer_section]

mod a {
    pub fn foo(xxx: u32) {
        super::_zzz(); // #break
    }
}

mod b {
    pub fn bar(yyy: u64) {
        super::_zzz(); // #break
    }
}

fn main() {
    a::foo(12345);
    b::bar(67890);
}

#[inline(never)]
fn _zzz() {}
