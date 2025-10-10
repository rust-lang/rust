//@ min-lldb-version: 1800

// LLDB (18.1+) now supports DW_TAG_variant_part, but there is some bug in either compiler or LLDB
// with memory layout of discriminant for this particular enum
// ensure that LLDB won't crash at least (like #57822).

//@ compile-flags:-g
//@ disable-gdb-pretty-printers

// === GDB TESTS ===================================================================================

// gdb-command:run
// gdb-command:print b
// gdb-check:$1 = coroutine_objects::main::{coroutine_env#0}::Unresumed{_ref__a: 0x[...]}
// gdb-command:continue
// gdb-command:print b
// gdb-check:$2 = coroutine_objects::main::{coroutine_env#0}::Suspend0{c: 6, d: 7, _ref__a: 0x[...]}
// gdb-command:continue
// gdb-command:print b
// gdb-check:$3 = coroutine_objects::main::{coroutine_env#0}::Suspend1{c: 7, d: 8, _ref__a: 0x[...]}
// gdb-command:continue
// gdb-command:print b
// gdb-check:$4 = coroutine_objects::main::{coroutine_env#0}::Returned{_ref__a: 0x[...]}

// === LLDB TESTS ==================================================================================

// lldb-command:run
// lldb-command:v b
// lldb-check:(coroutine_objects::main::{coroutine_env#0}) b = { value = { _ref__a = 0x[...] } $discr$ = [...] }

// === CDB TESTS ===================================================================================

// cdb-command: g
// cdb-command: dx b
// cdb-check: b                : Unresumed [Type: enum2$<coroutine_objects::main::coroutine_env$0>]
// cdb-check:    [+0x[...]] _ref__a          : 0x[...] : 5 [Type: int *]

// cdb-command: g
// cdb-command: dx b
// cdb-check: b                : Suspend0 [Type: enum2$<coroutine_objects::main::coroutine_env$0>]
// cdb-check:    [+0x[...]] c                : 6 [Type: int]
// cdb-check:    [+0x[...]] d                : 7 [Type: int]
// cdb-check:    [+0x[...]] _ref__a          : 0x[...] : 5 [Type: int *]

// cdb-command: g
// cdb-command: dx b
// cdb-check: b                : Suspend1 [Type: enum2$<coroutine_objects::main::coroutine_env$0>]
// cdb-check:    [+0x[...]] c                : 7 [Type: int]
// cdb-check:    [+0x[...]] d                : 8 [Type: int]
// cdb-check:    [+0x[...]] _ref__a          : 0x[...] : 6 [Type: int *]

// cdb-command: g
// cdb-command: dx b
// cdb-check: b                : Returned [Type: enum2$<coroutine_objects::main::coroutine_env$0>]
// cdb-check:    [+0x[...]] _ref__a          : 0x[...] : 6 [Type: int *]

#![feature(coroutines, coroutine_trait, stmt_expr_attributes)]

use std::ops::Coroutine;
use std::pin::Pin;

fn main() {
    let mut a = 5;
    let mut b = #[coroutine]
    || {
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

#[inline(never)]
fn _zzz() {
    ()
}
