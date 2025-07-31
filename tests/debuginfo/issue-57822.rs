// This test makes sure that the LLDB pretty printer does not throw an exception
// for nested closures and coroutines.

//@ min-lldb-version: 1800
//@ compile-flags:-g
//@ disable-gdb-pretty-printers

// === GDB TESTS ===================================================================================

// gdb-command:run

// gdb-command:print g
// gdb-check:$1 = issue_57822::main::{closure_env#1} {f: issue_57822::main::{closure_env#0} {x: 1}}

// gdb-command:print b
// gdb-check:$2 = issue_57822::main::{coroutine_env#3}::Unresumed{a: issue_57822::main::{coroutine_env#2}::Unresumed{y: 2}}

// === LLDB TESTS ==================================================================================

// lldb-command:run

// lldb-command:v g
// lldb-check:(issue_57822::main::{closure_env#1}) g = { f = { x = 1 } }

// lldb-command:v b
// lldb-check:(issue_57822::main::{coroutine_env#3}) b = { value = { a = { value = { y = 2 } $discr$ = '\x02' } } $discr$ = '\x02' }

#![feature(coroutines, coroutine_trait, stmt_expr_attributes)]

use std::ops::Coroutine;
use std::pin::Pin;

fn main() {
    let mut x = 1;
    let f = move || x;
    let g = move || f();

    let mut y = 2;
    let mut a = #[coroutine]
    move || {
        y += 1;
        yield;
    };
    let mut b = #[coroutine]
    move || {
        Pin::new(&mut a).resume(());
        yield;
    };

    zzz(); // #break
}

fn zzz() {
    ()
}
