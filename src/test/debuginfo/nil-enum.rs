// LLDB can't handle zero-sized values.
// ignore-lldb

// compile-flags:-g
// gdb-command:run

// gdb-command:print *first
// gdbg-check:$1 = {<No data fields>}
// gdbr-check:$1 = <error reading variable>

#![allow(unused_variables)]
#![feature(omit_gdb_pretty_printer_section)]
#![feature(maybe_uninit)]
#![omit_gdb_pretty_printer_section]

enum Void {}

fn main() {
    let first: *const Void = 1 as *const _;

    zzz(); // #break
}

fn zzz() {}
