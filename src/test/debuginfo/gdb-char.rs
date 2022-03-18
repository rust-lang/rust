// GDB got support for DW_ATE_UTF in 11.2, see
// https://sourceware.org/bugzilla/show_bug.cgi?id=28637.

// min-gdb-version: 11.2
// compile-flags: -g

// === GDB TESTS ===================================================================================

// gdb-command:run
// gdb-command:print ch
// gdb-check:$1 = 97 'a'
// gdb-command:whatis ch
// gdb-check:type = char
// gdb-command:whatis gdb_char::C
// gdb-check:type = char

#![allow(unused_variables)]
#![feature(omit_gdb_pretty_printer_section)]
#![omit_gdb_pretty_printer_section]

static mut C: char = 'a';

fn main() {
    let ch: char = 'a';

    zzz(); // #break

    let a = unsafe { (C, ) };
}

fn zzz() {()}
