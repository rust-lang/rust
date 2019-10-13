// This test makes sure that the LLDB pretty printer does not throw an exception
// for nested closures and generators.

// Require LLVM with DW_TAG_variant_part and a gdb that can read it.
// min-system-llvm-version: 8.0
// min-gdb-version: 8.2
// ignore-tidy-linelength

// compile-flags:-g

// === GDB TESTS ===================================================================================

// gdb-command:run

// gdb-command:print g
// gdb-check:$1 = issue_57822::main::closure-1 (issue_57822::main::closure-0 (1))

// gdb-command:print b
// gdb-check:$2 = issue_57822::main::generator-3 {__0: issue_57822::main::generator-2 {__0: 2, <<variant>>: {[...]}}, <<variant>>: {[...]}}

// === LLDB TESTS ==================================================================================

// lldb-command:run

// lldb-command:print g
// lldbg-check:(issue_57822::main::closure-1) $0 = closure-1(closure-0(1))

// lldb-command:print b
// lldbg-check:(issue_57822::main::generator-3) $1 = generator-3(generator-2(2))

#![feature(omit_gdb_pretty_printer_section, generators, generator_trait)]
#![omit_gdb_pretty_printer_section]

use std::ops::Generator;
use std::pin::Pin;

fn main() {
    let mut x = 1;
    let f = move || x;
    let g = move || f();

    let mut y = 2;
    let mut a = move || {
        y += 1;
        yield;
    };
    let mut b = move || {
        Pin::new(&mut a).resume();
        yield;
    };

    zzz(); // #break
}

fn zzz() { () }
