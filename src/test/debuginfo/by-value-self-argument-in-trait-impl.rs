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

#![feature(managed_boxes)]

// compile-flags:-g

// === GDB TESTS ===================================================================================

// gdb-command:rbreak zzz
// gdb-command:run

// gdb-command:finish
// gdb-command:print self
// gdb-check:$1 = 1111
// gdb-command:continue

// gdb-command:finish
// gdb-command:print self
// gdb-check:$2 = {x = 2222, y = 3333}
// gdb-command:continue

// gdb-command:finish
// gdb-command:print self
// gdb-check:$3 = {4444.5, 5555, 6666, 7777.5}
// gdb-command:continue

// gdb-command:finish
// gdb-command:print self->val
// gdb-check:$4 = 8888
// gdb-command:continue


// === LLDB TESTS ==================================================================================

// lldb-command:run

// lldb-command:print self
// lldb-check:[...]$0 = 1111
// lldb-command:continue

// lldb-command:print self
// lldb-check:[...]$1 = Struct { x: 2222, y: 3333 }
// lldb-command:continue

// lldb-command:print self
// lldb-check:[...]$2 = (4444.5, 5555, 6666, 7777.5)
// lldb-command:continue

// lldb-command:print self->val
// lldb-check:[...]$3 = 8888
// lldb-command:continue

use std::gc::{Gc, GC};

trait Trait {
    fn method(self) -> Self;
}

impl Trait for int {
    fn method(self) -> int {
        zzz(); // #break
        self
    }
}

struct Struct {
    x: uint,
    y: uint,
}

impl Trait for Struct {
    fn method(self) -> Struct {
        zzz(); // #break
        self
    }
}

impl Trait for (f64, int, int, f64) {
    fn method(self) -> (f64, int, int, f64) {
        zzz(); // #break
        self
    }
}

impl Trait for Gc<int> {
    fn method(self) -> Gc<int> {
        zzz(); // #break
        self
    }
}

fn main() {
    let _ = (1111 as int).method();
    let _ = Struct { x: 2222, y: 3333 }.method();
    let _ = (4444.5, 5555, 6666, 7777.5).method();
    let _ = (box(GC) 8888).method();
}

fn zzz() { () }
