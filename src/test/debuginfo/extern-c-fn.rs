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

// gdb-command:print s
// gdbg-check:$1 = [...]"abcd"
// gdbr-check:$1 = [...]"abcd\000"
// gdb-command:print len
// gdb-check:$2 = 20
// gdb-command:print local0
// gdb-check:$3 = 19
// gdb-command:print local1
// gdb-check:$4 = true
// gdb-command:print local2
// gdb-check:$5 = 20.5

// gdb-command:continue

// === LLDB TESTS ==================================================================================
// lldb-command:run

// lldb-command:print len
// lldb-check:[...]$0 = 20
// lldb-command:print local0
// lldb-check:[...]$1 = 19
// lldb-command:print local1
// lldb-check:[...]$2 = true
// lldb-command:print local2
// lldb-check:[...]$3 = 20.5

// lldb-command:continue

#![allow(unused_variables)]
#![allow(dead_code)]
#![feature(omit_gdb_pretty_printer_section)]
#![omit_gdb_pretty_printer_section]


#[no_mangle]
pub unsafe extern "C" fn fn_with_c_abi(s: *const u8, len: i32) -> i32 {
    let local0 = len - 1;
    let local1 = len > 2;
    let local2 = (len as f64) + 0.5;

    zzz(); // #break

    return 0;
}

fn main() {
    unsafe {
        fn_with_c_abi(b"abcd\0".as_ptr(), 20);
    }
}

#[inline(never)]
fn zzz() {()}
