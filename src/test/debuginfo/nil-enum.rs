// LLDB can't handle zero-sized values
// ignore-lldb


// compile-flags:-g
// gdb-command:run

// gdb-command:print first
// gdbg-check:$1 = {<No data fields>}
// gdbr-check:$1 = <error reading variable>

// gdb-command:print second
// gdbg-check:$2 = {<No data fields>}
// gdbr-check:$2 = <error reading variable>

#![allow(unused_variables)]
#![feature(omit_gdb_pretty_printer_section)]
#![omit_gdb_pretty_printer_section]

enum ANilEnum {}
enum AnotherNilEnum {}

// This test relies on gdbg printing the string "{<No data fields>}" for empty
// structs (which may change some time)
// The error from gdbr is expected since nil enums are not supposed to exist.
fn main() {
    unsafe {
        let first: ANilEnum = ::std::mem::zeroed();
        let second: AnotherNilEnum = ::std::mem::zeroed();

        zzz(); // #break
    }
}

fn zzz() {()}
