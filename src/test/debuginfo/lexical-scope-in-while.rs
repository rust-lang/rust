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

// FIRST ITERATION
// gdb-command:finish
// gdb-command:print x
// gdb-check:$1 = 0
// gdb-command:continue

// gdb-command:finish
// gdb-command:print x
// gdb-check:$2 = 1
// gdb-command:continue

// gdb-command:finish
// gdb-command:print x
// gdb-check:$3 = 101
// gdb-command:continue

// gdb-command:finish
// gdb-command:print x
// gdb-check:$4 = 101
// gdb-command:continue

// gdb-command:finish
// gdb-command:print x
// gdb-check:$5 = -987
// gdb-command:continue

// gdb-command:finish
// gdb-command:print x
// gdb-check:$6 = 101
// gdb-command:continue


// SECOND ITERATION
// gdb-command:finish
// gdb-command:print x
// gdb-check:$7 = 1
// gdb-command:continue

// gdb-command:finish
// gdb-command:print x
// gdb-check:$8 = 2
// gdb-command:continue

// gdb-command:finish
// gdb-command:print x
// gdb-check:$9 = 102
// gdb-command:continue

// gdb-command:finish
// gdb-command:print x
// gdb-check:$10 = 102
// gdb-command:continue

// gdb-command:finish
// gdb-command:print x
// gdb-check:$11 = -987
// gdb-command:continue

// gdb-command:finish
// gdb-command:print x
// gdb-check:$12 = 102
// gdb-command:continue

// gdb-command:finish
// gdb-command:print x
// gdb-check:$13 = 2
// gdb-command:continue


// === LLDB TESTS ==================================================================================

// lldb-command:run

// FIRST ITERATION
// lldb-command:print x
// lldb-check:[...]$0 = 0
// lldb-command:continue

// lldb-command:print x
// lldb-check:[...]$1 = 1
// lldb-command:continue

// lldb-command:print x
// lldb-check:[...]$2 = 101
// lldb-command:continue

// lldb-command:print x
// lldb-check:[...]$3 = 101
// lldb-command:continue

// lldb-command:print x
// lldb-check:[...]$4 = -987
// lldb-command:continue

// lldb-command:print x
// lldb-check:[...]$5 = 101
// lldb-command:continue


// SECOND ITERATION
// lldb-command:print x
// lldb-check:[...]$6 = 1
// lldb-command:continue

// lldb-command:print x
// lldb-check:[...]$7 = 2
// lldb-command:continue

// lldb-command:print x
// lldb-check:[...]$8 = 102
// lldb-command:continue

// lldb-command:print x
// lldb-check:[...]$9 = 102
// lldb-command:continue

// lldb-command:print x
// lldb-check:[...]$10 = -987
// lldb-command:continue

// lldb-command:print x
// lldb-check:[...]$11 = 102
// lldb-command:continue

// lldb-command:print x
// lldb-check:[...]$12 = 2
// lldb-command:continue


fn main() {

    let mut x = 0i;

    while x < 2 {
        zzz(); // #break
        sentinel();

        x += 1;
        zzz(); // #break
        sentinel();

        // Shadow x
        let x = x + 100;
        zzz(); // #break
        sentinel();

        // open scope within loop's top level scope
        {
            zzz(); // #break
            sentinel();

            let x = -987i;

            zzz(); // #break
            sentinel();
        }

        // Check that we get the x before the inner scope again
        zzz(); // #break
        sentinel();
    }

    zzz(); // #break
    sentinel();
}

fn zzz() {()}
fn sentinel() {()}
