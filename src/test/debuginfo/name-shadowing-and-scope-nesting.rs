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
// gdb-command:print x
// gdb-check:$1 = false
// gdb-command:print y
// gdb-check:$2 = true
// gdb-command:continue

// gdb-command:finish
// gdb-command:print x
// gdb-check:$3 = 10
// gdb-command:print y
// gdb-check:$4 = true
// gdb-command:continue

// gdb-command:finish
// gdb-command:print x
// gdb-check:$5 = 10.5
// gdb-command:print y
// gdb-check:$6 = 20
// gdb-command:continue

// gdb-command:finish
// gdb-command:print x
// gdb-check:$7 = true
// gdb-command:print y
// gdb-check:$8 = 2220
// gdb-command:continue

// gdb-command:finish
// gdb-command:print x
// gdb-check:$9 = 203203.5
// gdb-command:print y
// gdb-check:$10 = 2220
// gdb-command:continue

// gdb-command:finish
// gdb-command:print x
// gdb-check:$11 = 10.5
// gdb-command:print y
// gdb-check:$12 = 20
// gdb-command:continue


// === LLDB TESTS ==================================================================================

// lldb-command:run

// lldb-command:print x
// lldb-check:[...]$0 = false
// lldb-command:print y
// lldb-check:[...]$1 = true
// lldb-command:continue

// lldb-command:print x
// lldb-check:[...]$2 = 10
// lldb-command:print y
// lldb-check:[...]$3 = true
// lldb-command:continue

// lldb-command:print x
// lldb-check:[...]$4 = 10.5
// lldb-command:print y
// lldb-check:[...]$5 = 20
// lldb-command:continue

// lldb-command:print x
// lldb-check:[...]$6 = true
// lldb-command:print y
// lldb-check:[...]$7 = 2220
// lldb-command:continue

// lldb-command:print x
// lldb-check:[...]$8 = 203203.5
// lldb-command:print y
// lldb-check:[...]$9 = 2220
// lldb-command:continue

// lldb-command:print x
// lldb-check:[...]$10 = 10.5
// lldb-command:print y
// lldb-check:[...]$11 = 20
// lldb-command:continue

fn main() {
    let x = false;
    let y = true;

    zzz(); // #break
    sentinel();

    let x = 10i;

    zzz(); // #break
    sentinel();

    let x = 10.5f64;
    let y = 20i;

    zzz(); // #break
    sentinel();

    {
        let x = true;
        let y = 2220i;

        zzz(); // #break
        sentinel();

        let x = 203203.5f64;

        zzz(); // #break
        sentinel();
    }

    zzz(); // #break
    sentinel();
}

fn zzz() {()}
fn sentinel() {()}
