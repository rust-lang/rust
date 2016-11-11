// Copyright 2013-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(omit_gdb_pretty_printer_section)]
#![omit_gdb_pretty_printer_section]

// min-lldb-version: 310

// aux-build:cross_crate_spans.rs
extern crate cross_crate_spans;

// compile-flags:-g


// === GDB TESTS ===================================================================================

// gdb-command:break cross_crate_spans.rs:24
// gdb-command:run

// gdb-command:print result
// gdbg-check:$1 = {__0 = 17, __1 = 17}
// gdbr-check:$1 = (17, 17)
// gdb-command:print a_variable
// gdb-check:$2 = 123456789
// gdb-command:print another_variable
// gdb-check:$3 = 123456789.5
// gdb-command:continue

// gdb-command:print result
// gdbg-check:$4 = {__0 = 1212, __1 = 1212}
// gdbr-check:$4 = (1212, 1212)
// gdb-command:print a_variable
// gdb-check:$5 = 123456789
// gdb-command:print another_variable
// gdb-check:$6 = 123456789.5
// gdb-command:continue



// === LLDB TESTS ==================================================================================

// lldb-command:b cross_crate_spans.rs:24
// lldb-command:run

// lldb-command:print result
// lldb-check:[...]$0 = (17, 17)
// lldb-command:print a_variable
// lldb-check:[...]$1 = 123456789
// lldb-command:print another_variable
// lldb-check:[...]$2 = 123456789.5
// lldb-command:continue

// lldb-command:print result
// lldb-check:[...]$3 = (1212, 1212)
// lldb-command:print a_variable
// lldb-check:[...]$4 = 123456789
// lldb-command:print another_variable
// lldb-check:[...]$5 = 123456789.5
// lldb-command:continue


// This test makes sure that we can break in functions inlined from other crates.

fn main() {

    let _ = cross_crate_spans::generic_function(17u32);
    let _ = cross_crate_spans::generic_function(1212i16);

}
