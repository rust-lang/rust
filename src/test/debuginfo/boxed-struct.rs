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

// gdb-command:print *unique
// gdb-check:$1 = {x = 99, y = 999, z = 9999, w = 99999}

// gdb-command:print managed->val
// gdb-check:$2 = {x = 88, y = 888, z = 8888, w = 88888}

// gdb-command:print *unique_dtor
// gdb-check:$3 = {x = 77, y = 777, z = 7777, w = 77777}

// gdb-command:print managed_dtor->val
// gdb-check:$4 = {x = 33, y = 333, z = 3333, w = 33333}


// === LLDB TESTS ==================================================================================

// lldb-command:run

// lldb-command:print *unique
// lldb-check:[...]$0 = StructWithSomePadding { x: 99, y: 999, z: 9999, w: 99999 }

// lldb-command:print managed->val
// lldb-check:[...]$1 = StructWithSomePadding { x: 88, y: 888, z: 8888, w: 88888 }

// lldb-command:print *unique_dtor
// lldb-check:[...]$2 = StructWithDestructor { x: 77, y: 777, z: 7777, w: 77777 }

// lldb-command:print managed_dtor->val
// lldb-check:[...]$3 = StructWithDestructor { x: 33, y: 333, z: 3333, w: 33333 }

#![allow(unused_variable)]

use std::gc::GC;

struct StructWithSomePadding {
    x: i16,
    y: i32,
    z: i32,
    w: i64
}

struct StructWithDestructor {
    x: i16,
    y: i32,
    z: i32,
    w: i64
}

impl Drop for StructWithDestructor {
    fn drop(&mut self) {}
}

fn main() {

    let unique = box StructWithSomePadding { x: 99, y: 999, z: 9999, w: 99999 };
    let managed = box(GC) StructWithSomePadding { x: 88, y: 888, z: 8888, w: 88888 };

    let unique_dtor = box StructWithDestructor { x: 77, y: 777, z: 7777, w: 77777 };
    let managed_dtor = box(GC) StructWithDestructor { x: 33, y: 333, z: 3333, w: 33333 };

    zzz(); // #break
}

fn zzz() { () }
