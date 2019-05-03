// ignore-tidy-linelength

// Require LLVM with DW_TAG_variant_part and a gdb that can read it.
// min-system-llvm-version: 8.0
// min-gdb-version: 8.2

// compile-flags:-g

// === GDB TESTS ===================================================================================

// gdb-command:run
// gdb-command:print b
// gdb-check:$1 = generator_objects::main::generator {__0: 0x[...], <<variant>>: {__state: 0, 0: generator_objects::main::generator::Unresumed, 1: generator_objects::main::generator::Returned, 2: generator_objects::main::generator::Panicked, 3: generator_objects::main::generator::Suspend0 {[...]}, 4: generator_objects::main::generator::Suspend1 {[...]}}}
// gdb-command:continue
// gdb-command:print b
// gdb-check:$2 = generator_objects::main::generator {__0: 0x[...], <<variant>>: {__state: 3, 0: generator_objects::main::generator::Unresumed, 1: generator_objects::main::generator::Returned, 2: generator_objects::main::generator::Panicked, 3: generator_objects::main::generator::Suspend0 {c: 6, d: 7}, 4: generator_objects::main::generator::Suspend1 {[...]}}}
// gdb-command:continue
// gdb-command:print b
// gdb-check:$3 = generator_objects::main::generator {__0: 0x[...], <<variant>>: {__state: 4, 0: generator_objects::main::generator::Unresumed, 1: generator_objects::main::generator::Returned, 2: generator_objects::main::generator::Panicked, 3: generator_objects::main::generator::Suspend0 {[...]}, 4: generator_objects::main::generator::Suspend1 {c: 7, d: 8}}}
// gdb-command:continue
// gdb-command:print b
// gdb-check:$4 = generator_objects::main::generator {__0: 0x[...], <<variant>>: {__state: 1, 0: generator_objects::main::generator::Unresumed, 1: generator_objects::main::generator::Returned, 2: generator_objects::main::generator::Panicked, 3: generator_objects::main::generator::Suspend0 {[...]}, 4: generator_objects::main::generator::Suspend1 {[...]}}}

// === LLDB TESTS ==================================================================================

// lldb-command:run
// lldb-command:print b
// lldbg-check:(generator_objects::main::generator) $0 = generator(&0x[...])
// lldb-command:continue
// lldb-command:print b
// lldbg-check:(generator_objects::main::generator) $1 = generator(&0x[...])
// lldb-command:continue
// lldb-command:print b
// lldbg-check:(generator_objects::main::generator) $2 = generator(&0x[...])
// lldb-command:continue
// lldb-command:print b
// lldbg-check:(generator_objects::main::generator) $3 = generator(&0x[...])

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
    Pin::new(&mut b).resume();
    _zzz(); // #break
    Pin::new(&mut b).resume();
    _zzz(); // #break
    Pin::new(&mut b).resume();
    _zzz(); // #break
}

fn _zzz() {()}
