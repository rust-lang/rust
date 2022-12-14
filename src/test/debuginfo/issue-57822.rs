// This test makes sure that the LLDB pretty printer does not throw an exception
// for nested closures and generators.

// Require a gdb that can read DW_TAG_variant_part.
// min-gdb-version: 8.2

// compile-flags:-g

// === GDB TESTS ===================================================================================

// gdb-command:run

// gdb-command:print g
// gdb-check:$1 = issue_57822::main::{closure_env#1} {f: issue_57822::main::{closure_env#0} {x: 1}}

// gdb-command:print b
// gdb-check:$2 = issue_57822::main::{generator_env#3}::Unresumed{a: issue_57822::main::{generator_env#2}::Unresumed{y: 2}}

// === LLDB TESTS ==================================================================================

// lldb-command:run

// lldb-command:print g
// lldbg-check:(issue_57822::main::{closure_env#1}) $0 = { f = { x = 1 } }

// lldb-command:print b
// lldbg-check:(issue_57822::main::{generator_env#3}) $1 =

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
        Pin::new(&mut a).resume(());
        yield;
    };

    zzz(); // #break
}

fn zzz() {
    ()
}
