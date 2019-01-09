// min-lldb-version: 310
// ignore-gdb // Test temporarily ignored due to debuginfo tests being disabled, see PR 47155

// compile-flags:-g

// === GDB TESTS ===================================================================================

// gdb-command:run
// gdb-command:print a
// gdbg-check:$1 = {1, 2, 3}
// gdbr-check:$1 = [1, 2, 3]
// gdb-command:print vec::VECT
// gdbg-check:$2 = {4, 5, 6}
// gdbr-check:$2 = [4, 5, 6]


// === LLDB TESTS ==================================================================================

// lldb-command:run
// lldb-command:print a
// lldbg-check:[...]$0 = [1, 2, 3]
// lldbr-check:([i32; 3]) a = [1, 2, 3]

#![allow(unused_variables)]
#![feature(omit_gdb_pretty_printer_section)]
#![omit_gdb_pretty_printer_section]

static mut VECT: [i32; 3] = [1, 2, 3];

fn main() {
    let a = [1, 2, 3];

    unsafe {
        VECT[0] = 4;
        VECT[1] = 5;
        VECT[2] = 6;
    }

    zzz(); // #break
}

fn zzz() {()}
