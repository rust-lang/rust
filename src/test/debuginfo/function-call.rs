// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This test does not passed with gdb < 8.0. See #53497.
// min-gdb-version 8.0

// compile-flags:-g

// === GDB TESTS ===================================================================================

// gdb-command:run

// gdb-command:print fun(45, true)
// gdb-check:$1 = true
// gdb-command:print fun(444, false)
// gdb-check:$2 = false

// gdb-command:print r.get_x()
// gdb-check:$3 = 4

#![allow(dead_code, unused_variables)]

struct RegularStruct {
    x: i32
}

impl RegularStruct {
    fn get_x(&self) -> i32 {
        self.x
    }
}

fn main() {
    let _ = fun(4, true);
    let r = RegularStruct{x: 4};
    let _ = r.get_x();

    zzz(); // #break
}

fn fun(x: isize, y: bool) -> bool {
    y
}

fn zzz() { () }
