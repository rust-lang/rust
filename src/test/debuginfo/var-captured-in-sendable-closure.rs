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

// gdb-command:print constant
// gdb-check:$1 = 1
// gdb-command:print a_struct
// gdbg-check:$2 = {a = -2, b = 3.5, c = 4}
// gdbr-check:$2 = var_captured_in_sendable_closure::Struct {a: -2, b: 3.5, c: 4}
// gdb-command:print *owned
// gdb-check:$3 = 5
// gdb-command:continue

// gdb-command:print constant2
// gdb-check:$4 = 6
// gdb-command:continue

// === LLDB TESTS ==================================================================================

// lldb-command:run

// lldb-command:print constant
// lldb-check:[...]$0 = 1
// lldb-command:print a_struct
// lldb-check:[...]$1 = Struct { a: -2, b: 3.5, c: 4 }
// lldb-command:print *owned
// lldb-check:[...]$2 = 5

#![allow(unused_variables)]
#![feature(box_syntax)]
#![feature(omit_gdb_pretty_printer_section)]
#![omit_gdb_pretty_printer_section]

struct Struct {
    a: isize,
    b: f64,
    c: usize
}

fn main() {
    let constant = 1;

    let a_struct = Struct {
        a: -2,
        b: 3.5,
        c: 4
    };

    let owned: Box<_> = box 5;

    let closure = move || {
        zzz(); // #break
        do_something(&constant, &a_struct.a, &*owned);
    };

    closure();

    let constant2 = 6_usize;

    // The `self` argument of the following closure should be passed by value
    // to FnOnce::call_once(self, args), which gets translated a bit differently
    // than the regular case. Let's make sure this is supported too.
    let immedate_env = move || {
        zzz(); // #break
        return constant2;
    };

    immedate_env();
}

fn do_something(_: &isize, _:&isize, _:&isize) {

}

fn zzz() {()}
