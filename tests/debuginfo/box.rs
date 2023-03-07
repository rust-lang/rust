// min-lldb-version: 310

// compile-flags:-g

// === GDB TESTS ===================================================================================

// gdb-command:run

// gdb-command:print *a
// gdb-check:$1 = 1
// gdb-command:print *b
// gdbg-check:$2 = {__0 = 2, __1 = 3.5}
// gdbr-check:$2 = (2, 3.5)


// === LLDB TESTS ==================================================================================

// lldb-command:run
// lldb-command:print *a
// lldbg-check:[...]$0 = 1
// lldbr-check:(i32) *a = 1
// lldb-command:print *b
// lldbg-check:[...]$1 = { 0 = 2 1 = 3.5 }
// lldbr-check:((i32, f64)) *b = { 0 = 2 1 = 3.5 }

#![allow(unused_variables)]
#![feature(omit_gdb_pretty_printer_section)]
#![omit_gdb_pretty_printer_section]

fn main() {
    let a = Box::new(1);
    let b = Box::new((2, 3.5f64));

    zzz(); // #break
}

fn zzz() { () }
