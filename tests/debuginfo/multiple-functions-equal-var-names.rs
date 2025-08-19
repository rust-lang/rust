//@ compile-flags:-g
//@ disable-gdb-pretty-printers

// === GDB TESTS ===================================================================================

// gdb-command:run

// gdb-command:print abc
// gdb-check:$1 = 10101
// gdb-command:continue

// gdb-command:print abc
// gdb-check:$2 = 20202
// gdb-command:continue

// gdb-command:print abc
// gdb-check:$3 = 30303


// === LLDB TESTS ==================================================================================

// lldb-command:run

// lldb-command:v abc
// lldb-check:[...] 10101
// lldb-command:continue

// lldb-command:v abc
// lldb-check:[...] 20202
// lldb-command:continue

// lldb-command:v abc
// lldb-check:[...] 30303

#![allow(unused_variables)]

fn function_one() {
    let abc = 10101;
    zzz(); // #break
}

fn function_two() {
    let abc = 20202;
    zzz(); // #break
}


fn function_three() {
    let abc = 30303;
    zzz(); // #break
}


fn main() {
    function_one();
    function_two();
    function_three();
}

fn zzz() {()}
