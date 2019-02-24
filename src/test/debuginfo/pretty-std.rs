// ignore-windows failing on win32 bot
// ignore-freebsd: gdb package too new
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
// gdbg-check:$5 = Some = {8}
// gdbr-check:$5 = core::option::Option<i16>::Some(8)

// gdb-command: print none
// gdbg-check:$6 = None
// gdbr-check:$6 = core::option::Option<i64>::None

// gdb-command: print os_string
// gdb-check:$7 = "IAMA OS string"

// gdb-command: print some_string
// gdbg-check:$8 = {RUST$ENCODED$ENUM$0$None = Some = {"IAMA optional string!"}}
// gdbr-check:$8 = core::option::Option<alloc::string::String>::Some("IAMA optional string!")

// gdb-command: set print elements 5
// gdb-command: print some_string
// gdbg-check:$9 = {RUST$ENCODED$ENUM$0$None = Some = {"IAMA "...}}
// gdbr-check:$9 = core::option::Option<alloc::string::String>::Some("IAMA "...)

// gdb-command: print empty_str
// gdb-check:$10 = ""

// === LLDB TESTS ==================================================================================

// lldb-command: run

// lldb-command: fr v slice
// lldb-check:[...]slice = &[0, 1, 2, 3]

// lldb-command: fr v vec
// lldb-check:[...]vec = vec![4, 5, 6, 7]

// lldb-command: fr v str_slice
// lldb-check:[...]str_slice = "IAMA string slice!"

// lldb-command: fr v string
// lldb-check:[...]string = "IAMA string!"

// FIXME #58492
// lldb-command: fr v some
// lldb-check:[...]some = Option<i16> { }

// FIXME #58492
// lldb-command: fr v none
// lldb-check:[...]none = Option<i64> { }

// lldb-command: fr v empty_str
// lldb-check:[...]empty_str = ""

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
    let os_string = OsString::from("IAMA OS string");

    // Option
    let some = Some(8i16);
    let none: Option<i64> = None;

    let some_string = Some("IAMA optional string!".to_owned());

    let empty_str = "";

    zzz(); // #break
}

fn zzz() { () }
