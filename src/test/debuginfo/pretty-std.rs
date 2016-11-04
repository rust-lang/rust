// Copyright 2013-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-windows failing on win32 bot
// ignore-freebsd: gdb package too new
// ignore-android: FIXME(#10381)
// compile-flags:-g
// min-gdb-version 7.7
// min-lldb-version: 310

// === GDB TESTS ===================================================================================

// gdb-command: run

// gdb-command: print slice
// gdb-check:$1 = &[i32](len: 4) = {0, 1, 2, 3}

// gdb-command: print vec
// gdb-check:$2 = Vec<u64>(len: 4, cap: [...]) = {4, 5, 6, 7}

// gdb-command: print str_slice
// gdb-check:$3 = "IAMA string slice!"

// gdb-command: print string
// gdb-check:$4 = "IAMA string!"

// gdb-command: print some
// gdb-check:$5 = Some = {8}

// gdb-command: print none
// gdbg-check:$6 = None
// gdbr-check:$6 = core::option::Option::None


// === LLDB TESTS ==================================================================================

// lldb-command: run

// lldb-command: print slice
// lldb-check:[...]$0 = &[0, 1, 2, 3]

// lldb-command: print vec
// lldb-check:[...]$1 = vec![4, 5, 6, 7]

// lldb-command: print str_slice
// lldb-check:[...]$2 = "IAMA string slice!"

// lldb-command: print string
// lldb-check:[...]$3 = "IAMA string!"

// lldb-command: print some
// lldb-check:[...]$4 = Some(8)

// lldb-command: print none
// lldb-check:[...]$5 = None


#![allow(unused_variables)]

fn main() {

    // &[]
    let slice: &[i32] = &[0, 1, 2, 3];

    // Vec
    let vec = vec![4u64, 5, 6, 7];

    // &str
    let str_slice = "IAMA string slice!";

    // String
    let string = "IAMA string!".to_string();

    // Option
    let some = Some(8i16);
    let none: Option<i64> = None;

    zzz(); // #break
}

fn zzz() { () }
