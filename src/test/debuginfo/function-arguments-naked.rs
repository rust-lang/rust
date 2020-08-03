// min-lldb-version: 310

// We have to ignore android because of this issue:
// https://github.com/rust-lang/rust/issues/74847
// ignore-android

// compile-flags:-g

// === GDB TESTS ===================================================================================

// gdb-command:run

// gdb-command:info args
// gdb-check:No arguments.
// gdb-command:continue

// === LLDB TESTS ==================================================================================

// lldb-command:run

// lldb-command:frame variable
// lldbg-check:(unsigned long) = 111 (unsigned long) = 222
// lldbr-check:(unsigned long) = 111 (unsigned long) = 222
// lldb-command:continue


#![feature(naked_functions)]
#![feature(omit_gdb_pretty_printer_section)]
#![omit_gdb_pretty_printer_section]

fn main() {
    naked(111, 222);
}

#[naked]
fn naked(x: usize, y: usize) {
    zzz(); // #break
}

fn zzz() { () }
