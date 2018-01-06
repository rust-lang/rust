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

// == Test [gdb|lldb]-[command|check] are parsed correctly ===
// should-fail
// compile-flags:-g

// === GDB TESTS ===================================================================================

// gdb-command: run

// gdb-command: print x
// gdb-check:$1 = 5

// === LLDB TESTS ==================================================================================

// lldb-command:run

// lldb-command:print x
// lldb-check:[...]$0 = 5

fn main() {
    let x = 1;

    zzz(); // #break
}

fn zzz() {()}

