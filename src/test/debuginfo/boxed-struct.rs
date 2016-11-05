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

// gdb-command:print *unique
// gdbg-check:$1 = {x = 99, y = 999, z = 9999, w = 99999}
// gdbr-check:$1 = boxed_struct::StructWithSomePadding {x: 99, y: 999, z: 9999, w: 99999}

// gdb-command:print *unique_dtor
// gdbg-check:$2 = {x = 77, y = 777, z = 7777, w = 77777}
// gdbr-check:$2 = boxed_struct::StructWithDestructor {x: 77, y: 777, z: 7777, w: 77777}


// === LLDB TESTS ==================================================================================

// lldb-command:run

// lldb-command:print *unique
// lldb-check:[...]$0 = StructWithSomePadding { x: 99, y: 999, z: 9999, w: 99999 }

// lldb-command:print *unique_dtor
// lldb-check:[...]$1 = StructWithDestructor { x: 77, y: 777, z: 7777, w: 77777 }

#![allow(unused_variables)]
#![feature(box_syntax)]
#![feature(omit_gdb_pretty_printer_section)]
#![omit_gdb_pretty_printer_section]

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

    let unique: Box<_> = box StructWithSomePadding { x: 99, y: 999, z: 9999, w: 99999 };

    let unique_dtor: Box<_> = box StructWithDestructor { x: 77, y: 777, z: 7777, w: 77777 };
    zzz(); // #break
}

fn zzz() { () }
