// min-lldb-version: 310

// compile-flags:-g
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

// lldb-command:print string1.length
// lldbg-check:[...]$0 = 48
// lldbr-check:(usize) length = 48
// lldb-command:print string2.length
// lldbg-check:[...]$1 = 49
// lldbr-check:(usize) length = 49
// lldb-command:print string3.length
// lldbg-check:[...]$2 = 50
// lldbr-check:(usize) length = 50

// lldb-command:continue

#![allow(unused_variables)]
#![feature(omit_gdb_pretty_printer_section)]
#![omit_gdb_pretty_printer_section]

// This test case makes sure that debug info does not ICE when include_str is
// used multiple times (see issue #11322).

fn main() {
    let string1 = include_str!("text-to-include-1.txt");
    let string2 = include_str!("text-to-include-2.txt");
    let string3 = include_str!("text-to-include-3.txt");

    zzz(); // #break
}

fn zzz() {()}
