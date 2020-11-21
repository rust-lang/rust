// ignore-freebsd: gdb package too new
// only-cdb // "Temporarily" ignored on GDB/LLDB due to debuginfo tests being disabled, see PR 47155
// ignore-android: FIXME(#10381)
// compile-flags:-g
// min-gdb-version: 7.7
// min-lldb-version: 310

// === GDB TESTS ===================================================================================

// gdb-command: run

// gdb-command: print slice
// gdb-check:$1 = &[i32](len: 4) = {0, 1, 2, 3}

// gdb-command: print vec
// gdb-check:$2 = Vec<u64, alloc::alloc::Global>(len: 4, cap: [...]) = {4, 5, 6, 7}

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

// lldb-command: print os_string
// lldb-check:[...]$6 = "IAMA OS string 😃"[...]


// === CDB TESTS ==================================================================================

// cdb-command: g

// cdb-command: dx slice,d
// cdb-check:slice,d [...]
// NOTE: While slices have a .natvis entry that works in VS & VS Code, it fails in CDB 10.0.18362.1

// cdb-command: dx vec,d
// cdb-check:vec,d [...] : { size=4 } [Type: [...]::Vec<u64, alloc::alloc::Global>]
// cdb-check:    [size]           : 4 [Type: [...]]
// cdb-check:    [capacity]       : [...] [Type: [...]]
// cdb-check:    [0]              : 4 [Type: unsigned __int64]
// cdb-check:    [1]              : 5 [Type: unsigned __int64]
// cdb-check:    [2]              : 6 [Type: unsigned __int64]
// cdb-check:    [3]              : 7 [Type: unsigned __int64]

// cdb-command: dx str_slice
// cdb-check:str_slice [...]
// NOTE: While string slices have a .natvis entry that works in VS & VS Code, it fails in CDB

// cdb-command: dx string
// cdb-check:string           : "IAMA string!" [Type: [...]::String]
// cdb-check:    [<Raw View>]     [Type: [...]::String]
// cdb-check:    [size]           : 0xc [Type: [...]]
// cdb-check:    [capacity]       : 0xc [Type: [...]]
// cdb-check:    [0]              : 73 'I' [Type: char]
// cdb-check:    [1]              : 65 'A' [Type: char]
// cdb-check:    [2]              : 77 'M' [Type: char]
// cdb-check:    [3]              : 65 'A' [Type: char]
// cdb-check:    [4]              : 32 ' ' [Type: char]
// cdb-check:    [5]              : 115 's' [Type: char]
// cdb-check:    [6]              : 116 't' [Type: char]
// cdb-check:    [7]              : 114 'r' [Type: char]
// cdb-check:    [8]              : 105 'i' [Type: char]
// cdb-check:    [9]              : 110 'n' [Type: char]
// cdb-check:    [10]             : 103 'g' [Type: char]
// cdb-check:    [11]             : 33 '!' [Type: char]

// cdb-command: dx os_string
// cdb-check:os_string        [Type: [...]::OsString]
// NOTE: OsString doesn't have a .natvis entry yet.

// cdb-command: dx some
// cdb-check:some             : { Some 8 } [Type: [...]::Option<i16>]
// cdb-command: dx none
// cdb-check:none             : { None } [Type: [...]::Option<i64>]
// cdb-command: dx some_string
// cdb-check:some_string      : { Some "IAMA optional string!" } [[...]::Option<[...]::String>]

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
