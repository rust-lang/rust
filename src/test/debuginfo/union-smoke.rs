// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
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
// gdb-command:print u
// gdb-check:$1 = {a = 11 '\v', b = 11}
// gdb-command:print union_smoke::SU
// gdb-check:$2 = {a = 10 '\n', b = 10}

// === LLDB TESTS ==================================================================================

// lldb-command:run
// lldb-command:print a
// lldb-check:[...]$0 = {a = 11 '\v', b = 11}
// lldb-command:print union_smoke::SU
// lldb-check:[...]$1 = {a = 10 '\n', b = 10}

#![allow(unused)]
#![feature(omit_gdb_pretty_printer_section)]
#![omit_gdb_pretty_printer_section]
#![feature(untagged_unions)]

union U {
    a: u8,
    b: u64,
}

static SU: U = U { a: 10 };

fn main() {
    let u = U { b: 11 };

    zzz(); // #break
}

fn zzz() {()}
