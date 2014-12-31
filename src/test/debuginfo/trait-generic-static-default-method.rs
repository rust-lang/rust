// ignore-test

// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags:-g

// === GDB TESTS ===================================================================================

// gdb-command:run

// gdb-command:print arg1
// gdb-check:$1 = 1000
// gdb-command:print arg2
// gdb-check:$2 = 0.5
// gdb-command:continue

// gdb-command:print arg1
// gdb-check:$3 = 2000
// gdb-command:print *arg2
// gdb-check:$4 = {1, 2, 3}
// gdb-command:continue


// === LLDB TESTS ==================================================================================

// lldb-command:run

// lldb-command:print arg1
// lldb-check:[...]$0 = 1000
// lldb-command:print arg2
// lldb-check:[...]$1 = 0.5
// lldb-command:continue

// lldb-command:print arg1
// lldb-check:[...]$2 = 2000
// lldb-command:print *arg2
// lldb-check:[...]$3 = (1, 2, 3)
// lldb-command:continue

#![omit_gdb_pretty_printer_section]

struct Struct {
    x: int
}

trait Trait {
    fn generic_static_default_method<T>(arg1: int, arg2: T) -> int {
        zzz(); // #break
        arg1
    }
}

impl Trait for Struct {}

fn main() {

    // Is this really how to use these?
    Trait::generic_static_default_method::<Struct, float>(1000, 0.5);
    Trait::generic_static_default_method::<Struct, &(int, int, int)>(2000, &(1, 2, 3));

}

fn zzz() {()}
