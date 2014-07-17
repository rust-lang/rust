// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-android: FIXME(#10381)

// compile-flags:-g

// === GDB TESTS ===================================================================================

// gdb-command:rbreak zzz
// gdb-command:run
// gdb-command:finish

// gdb-command:print constant
// gdb-check:$1 = 1
// gdb-command:print a_struct
// gdb-check:$2 = {a = -2, b = 3.5, c = 4}
// gdb-command:print *owned
// gdb-check:$3 = 5


// === LLDB TESTS ==================================================================================

// lldb-command:run

// lldb-command:print constant
// lldb-check:[...]$0 = 1
// lldb-command:print a_struct
// lldb-check:[...]$1 = Struct { a: -2, b: 3.5, c: 4 }
// lldb-command:print *owned
// lldb-check:[...]$2 = 5

#![allow(unused_variable)]

struct Struct {
    a: int,
    b: f64,
    c: uint
}

fn main() {
    let constant = 1;

    let a_struct = Struct {
        a: -2,
        b: 3.5,
        c: 4
    };

    let owned = box 5;

    let closure: proc() = proc() {
        zzz(); // #break
        do_something(&constant, &a_struct.a, &*owned);
    };

    closure();
}

fn do_something(_: &int, _:&int, _:&int) {

}

fn zzz() {()}
