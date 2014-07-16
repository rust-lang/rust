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
// gdb-command:continue

// gdb-command:finish
// gdb-command:print x
// gdb-check:$2 = false
// gdb-command:continue

// gdb-command:finish
// gdb-command:print x
// gdb-check:$3 = 1000
// gdb-command:continue

// gdb-command:finish
// gdb-command:print x
// gdb-check:$4 = 2.5
// gdb-command:continue

// gdb-command:finish
// gdb-command:print x
// gdb-check:$5 = true
// gdb-command:continue

// gdb-command:finish
// gdb-command:print x
// gdb-check:$6 = false
// gdb-command:continue


// === LLDB TESTS ==================================================================================

// lldb-command:run

// lldb-command:print x
// lldb-check:[...]$0 = false
// lldb-command:continue

// lldb-command:print x
// lldb-check:[...]$1 = false
// lldb-command:continue

// lldb-command:print x
// lldb-check:[...]$2 = 1000
// lldb-command:continue

// lldb-command:print x
// lldb-check:[...]$3 = 2.5
// lldb-command:continue

// lldb-command:print x
// lldb-check:[...]$4 = true
// lldb-command:continue

// lldb-command:print x
// lldb-check:[...]$5 = false
// lldb-command:continue

fn main() {

    let x = false;

    zzz(); // #break
    sentinel();

    let unique_closure: proc(int) = proc(x) {
        zzz(); // #break
        sentinel();

        let x = 2.5f64;

        zzz(); // #break
        sentinel();

        let x = true;

        zzz(); // #break
        sentinel();
    };

    zzz(); // #break
    sentinel();

    unique_closure(1000);

    zzz(); // #break
    sentinel();
}

fn zzz() {()}
fn sentinel() {()}
