// min-lldb-version: 310

// compile-flags:-g

// === GDB TESTS ===================================================================================

// gdb-command:run
// gdb-command:print a
// gdb-check:$1 = 5

// === LLDB TESTS ==================================================================================

// lldb-command:run
// lldb-command:print a
// lldbg-check:(int) $0 = 5
// lldbr-check:(int) a = 5

#![feature(omit_gdb_pretty_printer_section, generators, generator_trait, pin)]
#![omit_gdb_pretty_printer_section]

use std::ops::Generator;
use std::pin::Pin;

fn main() {
    let mut a = 5;
    let mut b = || {
        yield;
        _zzz(); // #break
        a = 6;
    };
    Pin::new(&mut b).resume();
    Pin::new(&mut b).resume();
    _zzz(); // #break
}

fn _zzz() {()}
