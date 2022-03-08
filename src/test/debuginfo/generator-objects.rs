// Require a gdb that can read DW_TAG_variant_part.
// min-gdb-version: 8.2

// LLDB without native Rust support cannot read DW_TAG_variant_part,
// so it prints nothing for generators. But those tests are kept to
// ensure that LLDB won't crash at least (like #57822).

// compile-flags:-g

// === GDB TESTS ===================================================================================

// gdb-command:run
// gdb-command:print b
// gdb-check:$1 = generator_objects::main::{generator_env#0}::Unresumed{_ref__a: 0x[...]}
// gdb-command:continue
// gdb-command:print b
// gdb-check:$2 = generator_objects::main::{generator_env#0}::Suspend0{c: 6, d: 7, _ref__a: 0x[...]}
// gdb-command:continue
// gdb-command:print b
// gdb-check:$3 = generator_objects::main::{generator_env#0}::Suspend1{c: 7, d: 8, _ref__a: 0x[...]}
// gdb-command:continue
// gdb-command:print b
// gdb-check:$4 = generator_objects::main::{generator_env#0}::Returned{_ref__a: 0x[...]}

// === LLDB TESTS ==================================================================================

// lldb-command:run
// lldb-command:print b
// lldbg-check:(generator_objects::main::{generator_env#0}) $0 =
// lldb-command:continue
// lldb-command:print b
// lldbg-check:(generator_objects::main::{generator_env#0}) $1 =
// lldb-command:continue
// lldb-command:print b
// lldbg-check:(generator_objects::main::{generator_env#0}) $2 =
// lldb-command:continue
// lldb-command:print b
// lldbg-check:(generator_objects::main::{generator_env#0}) $3 =

#![feature(omit_gdb_pretty_printer_section, generators, generator_trait)]
#![omit_gdb_pretty_printer_section]

use std::ops::Generator;
use std::pin::Pin;

fn main() {
    let mut a = 5;
    let mut b = || {
        let mut c = 6;
        let mut d = 7;

        yield;
        a += 1;
        c += 1;
        d += 1;

        yield;
        println!("{} {} {}", a, c, d);
    };
    _zzz(); // #break
    Pin::new(&mut b).resume(());
    _zzz(); // #break
    Pin::new(&mut b).resume(());
    _zzz(); // #break
    Pin::new(&mut b).resume(());
    _zzz(); // #break
}

fn _zzz() {
    ()
}
