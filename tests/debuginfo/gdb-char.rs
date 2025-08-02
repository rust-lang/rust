// GDB got support for DW_ATE_UTF in 11.2, see
// https://sourceware.org/bugzilla/show_bug.cgi?id=28637.

//@ min-gdb-version: 11.2
//@ compile-flags: -g
//@ disable-gdb-pretty-printers

// === GDB TESTS ===================================================================================

// gdb-command:run
// gdb-command:print ch
// gdb-check:$1 = 97 'a'

#![allow(unused_variables)]

fn main() {
    let ch: char = 'a';

    zzz(); // #break
}

fn zzz() {()}
