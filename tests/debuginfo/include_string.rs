//@ ignore-gdb-version: 15.0 - 99.0
// ^ test temporarily disabled as it fails under gdb 15

//@ compile-flags:-g
//@ disable-gdb-pretty-printers
// gdb-command:run
// gdb-command:print string1.length
// gdb-check:$1 = 48
// gdb-command:print string2.length
// gdb-check:$2 = 49
// gdb-command:print string3.length
// gdb-check:$3 = 50
// gdb-command:continue


// === LLDB TESTS ==================================================================================

// lldb-command:run

// lldb-command:v string1.length
// lldb-check:[...] 48
// lldb-command:v string2.length
// lldb-check:[...] 49
// lldb-command:v string3.length
// lldb-check:[...] 50

// lldb-command:continue

#![allow(unused_variables)]

// This test case makes sure that debug info does not ICE when include_str is
// used multiple times (see issue #11322).

fn main() {
    let string1 = include_str!("text-to-include-1.txt");
    let string2 = include_str!("text-to-include-2.txt");
    let string3 = include_str!("text-to-include-3.txt");

    zzz(); // #break
}

fn zzz() {()}
