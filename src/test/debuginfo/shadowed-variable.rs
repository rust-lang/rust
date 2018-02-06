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

// gdb-command:print x
// gdb-check:$1 = false
// gdb-command:print y
// gdb-check:$2 = true
// gdb-command:continue

// gdb-command:print x
// gdb-check:$3 = 10
// gdb-command:print y
// gdb-check:$4 = true
// gdb-command:continue

// gdb-command:print x
// gdb-check:$5 = 10.5
// gdb-command:print y
// gdb-check:$6 = 20
// gdb-command:continue

// gdb-command:print x
// gdb-check:$7 = 10.5
// gdb-command:print y
// gdb-check:$8 = 20
// gdb-command:continue

// gdb-command:print x
// gdb-check:$9 = 11.5
// gdb-command:print y
// gdb-check:$10 = 20
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
// lldb-check:[...]$6 = 10.5
// lldb-command:print y
// lldb-check:[...]$7 = 20
// lldb-command:continue

// lldb-command:print x
// lldb-check:[...]$8 = 11.5
// lldb-command:print y
// lldb-check:[...]$9 = 20
// lldb-command:continue

#![feature(omit_gdb_pretty_printer_section)]
#![omit_gdb_pretty_printer_section]

fn main() {
    let x = false;
    let y = true;

    zzz(); // #break
    sentinel();

    let x = 10;

    zzz(); // #break
    sentinel();

    let x = 10.5f64;
    let y = 20;

    zzz(); // #break
    sentinel();

    let x = {
        zzz(); // #break
        sentinel();
        11.5
    };

    zzz(); // #break
    sentinel()
}

fn zzz() {()}
fn sentinel() {()}
