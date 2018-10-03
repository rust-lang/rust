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

// gdb-command:print x
// gdb-check:$1 = 0.5
// gdb-command:print y
// gdb-check:$2 = 10
// gdb-command:continue

// gdb-command:print *x
// gdb-check:$3 = 29
// gdb-command:print *y
// gdb-check:$4 = 110
// gdb-command:continue


// === LLDB TESTS ==================================================================================

// lldb-command:run

// lldb-command:print x
// lldbg-check:[...]$0 = 0.5
// lldbr-check:(f64) x = 0.5
// lldb-command:print y
// lldbg-check:[...]$1 = 10
// lldbr-check:(i32) y = 10
// lldb-command:continue

// lldb-command:print *x
// lldbg-check:[...]$2 = 29
// lldbr-check:(i32) *x = 29
// lldb-command:print *y
// lldbg-check:[...]$3 = 110
// lldbr-check:(i32) *y = 110
// lldb-command:continue

#![feature(box_syntax)]
#![feature(omit_gdb_pretty_printer_section)]
#![omit_gdb_pretty_printer_section]

fn some_generic_fun<T1, T2>(a: T1, b: T2) -> (T2, T1) {

    let closure = |x, y| {
        zzz(); // #break
        (y, x)
    };

    closure(a, b)
}

fn main() {
    some_generic_fun(0.5f64, 10);
    some_generic_fun(&29, Box::new(110));
}

fn zzz() { () }
