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

// gdb-command:print a
// gdb-check:$1 = 10101
// gdb-command:continue

// gdb-command:finish
// gdb-command:print b
// gdb-check:$2 = 20202
// gdb-command:continue

// gdb-command:finish
// gdb-command:print c
// gdb-check:$3 = 30303


// === LLDB TESTS ==================================================================================

// lldb-command:run

// lldb-command:print a
// lldb-check:[...]$0 = 10101
// lldb-command:continue

// lldb-command:print b
// lldb-check:[...]$1 = 20202
// lldb-command:continue

// lldb-command:print c
// lldb-check:[...]$2 = 30303

#![allow(unused_variable)]

fn function_one() {
    let a = 10101i;
    zzz(); // #break
}

fn function_two() {
    let b = 20202i;
    zzz(); // #break
}


fn function_three() {
    let c = 30303i;
    zzz(); // #break
}


fn main() {
    function_one();
    function_two();
    function_three();
}

fn zzz() {()}
