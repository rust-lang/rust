// min-lldb-version: 310

// compile-flags:-g

// === GDB TESTS ===================================================================================

// gdb-command:run

// gdb-command:print a
// gdb-check:$1 = 10101
// gdb-command:continue

// gdb-command:print b
// gdb-check:$2 = 20202
// gdb-command:continue

// gdb-command:print c
// gdb-check:$3 = 30303


// === LLDB TESTS ==================================================================================

// lldb-command:run

// lldb-command:print a
// lldb-check:[...]$0 = 10101
// lldb-command:continue

// lldb-command:print b
// lldb-check:[...]$1 = 20202
// lldb-command:continue

// lldb-command:print c
// lldb-check:[...]$2 = 30303

#![allow(unused_variables)]
#![feature(omit_gdb_pretty_printer_section)]
#![omit_gdb_pretty_printer_section]

fn function_one() {
    let a = 10101;
    zzz(); // #break
}

fn function_two() {
    let b = 20202;
    zzz(); // #break
}


fn function_three() {
    let c = 30303;
    zzz(); // #break
}


fn main() {
    function_one();
    function_two();
    function_three();
}

fn zzz() {()}
