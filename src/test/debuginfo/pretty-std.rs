// ignore-windows failing on win32 bot
// ignore-freebsd: gdb package too new
// ignore-test // Test temporarily ignored due to debuginfo tests being disabled, see PR 47155
// ignore-android: FIXME(#10381)
// compile-flags:-g
// min-gdb-version 7.7
// min-lldb-version: 310

// === GDB TESTS ===================================================================================

// gdb-command: run

// gdb-command: print slice
// gdb-check:$1 = &[i32](len: 4) = {0, 1, 2, 3}

// gdb-command: print vec
// gdb-check:$2 = Vec<u64>(len: 4, cap: [...]) = {4, 5, 6, 7}

// gdb-command: print str_slice
// gdb-check:$3 = "IAMA string slice!"

// gdb-command: print string
// gdb-check:$4 = "IAMA string!"

// gdb-command: print some
// gdb-check:$5 = Some = {8}

// gdb-command: print none
// gdbg-check:$6 = None
// gdbr-check:$6 = core::option::Option::None

// gdb-command: print os_string
// gdb-check:$7 = "IAMA OS string 😃"

// gdb-command: print some_string
// gdb-check:$8 = Some = {"IAMA optional string!"}

// gdb-command: set print length 5
// gdb-command: print some_string
// gdb-check:$8 = Some = {"IAMA "...}


// === LLDB TESTS ==================================================================================

// lldb-command: run

// lldb-command: print slice
// lldb-check:[...]$0 = &[0, 1, 2, 3]

// lldb-command: print vec
// lldb-check:[...]$1 = vec![4, 5, 6, 7]

// lldb-command: print str_slice
// lldb-check:[...]$2 = "IAMA string slice!"

// lldb-command: print string
// lldb-check:[...]$3 = "IAMA string!"

// lldb-command: print some
// lldb-check:[...]$4 = Some(8)

// lldb-command: print none
// lldb-check:[...]$5 = None


#![allow(unused_variables)]
use std::ffi::OsString;


fn main() {

    // &[]
    let slice: &[i32] = &[0, 1, 2, 3];

    // Vec
    let vec = vec![4u64, 5, 6, 7];

    // &str
    let str_slice = "IAMA string slice!";

    // String
    let string = "IAMA string!".to_string();

    // OsString
    let os_string = OsString::from("IAMA OS string \u{1F603}");

    // Option
    let some = Some(8i16);
    let none: Option<i64> = None;

    let some_string = Some("IAMA optional string!".to_owned());

    zzz(); // #break
}

fn zzz() { () }
