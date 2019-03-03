// min-lldb-version: 310

// compile-flags:-g

// === GDB TESTS ===================================================================================

// gdb-command:run
// gdb-command:print a
// gdb-check:$1 = 5
// gdb-command:print d
// gdb-check:$2 = 6

// === LLDB TESTS ==================================================================================

// lldb-command:run
// lldb-command:print a
// lldbg-check:(int) $0 = 5
// lldbr-check:(int) a = 5
// lldb-command:print d
// lldbg-check:(int) $1 = 6
// lldbr-check:(int) d = 6

#![feature(omit_gdb_pretty_printer_section, generators, generator_trait)]
#![omit_gdb_pretty_printer_section]

use std::ops::Generator;
use std::pin::Pin;

fn main() {
    let mut a = 5;
    let mut b = || {
        let d = 6;
        yield;
        _zzz(); // #break
        a = d;
    };
    Pin::new(&mut b).resume();
    Pin::new(&mut b).resume();
    _zzz(); // #break
}

fn _zzz() {()}
