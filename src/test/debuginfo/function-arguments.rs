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
// gdb-check:$1 = 111102
// gdb-command:print y
// gdb-check:$2 = true

// gdb-command:continue
// gdb-command:finish

// gdb-command:print a
// gdb-check:$3 = 2000
// gdb-command:print b
// gdb-check:$4 = 3000


// === LLDB TESTS ==================================================================================

// lldb-command:run

// lldb-command:print x
// lldb-check:[...]$0 = 111102
// lldb-command:print y
// lldb-check:[...]$1 = true
// lldb-command:continue

// lldb-command:print a
// lldb-check:[...]$2 = 2000
// lldb-command:print b
// lldb-check:[...]$3 = 3000
// lldb-command:continue

fn main() {

    fun(111102, true);
    nested(2000, 3000);

    fn nested(a: i32, b: i64) -> (i32, i64) {
        zzz(); // #break
        (a, b)
    }
}

fn fun(x: int, y: bool) -> (int, bool) {
    zzz(); // #break

    (x, y)
}

fn zzz() { () }
