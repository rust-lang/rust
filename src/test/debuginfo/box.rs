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

// gdb-command:set print pretty off
// gdb-command:rbreak zzz
// gdb-command:run
// gdb-command:finish
// gdb-command:print *a
// gdb-check:$1 = 1
// gdb-command:print *b
// gdb-check:$2 = {2, 3.5}
// gdb-command:print c->val
// gdb-check:$3 = 4
// gdb-command:print d->val
// gdb-check:$4 = false


// === LLDB TESTS ==================================================================================

// lldb-command:run
// lldb-command:print *a
// lldb-check:[...]$0 = 1
// lldb-command:print *b
// lldb-check:[...]$1 = (2, 3.5)
// lldb-command:print c->val
// lldb-check:[...]$2 = 4
// lldb-command:print d->val
// lldb-check:[...]$3 = false

#![feature(managed_boxes)]
#![allow(unused_variable)]

use std::gc::GC;

fn main() {
    let a = box 1i;
    let b = box() (2i, 3.5f64);
    let c = box(GC) 4i;
    let d = box(GC) false;

    zzz(); // #break
}

fn zzz() { () }
