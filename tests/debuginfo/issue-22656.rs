// This test makes sure that the LLDB pretty printer does not throw an exception
// when trying to handle a Vec<> or anything else that contains zero-sized
// fields.

//@ min-lldb-version: 310
//@ ignore-gdb

//@ compile-flags:-g

// === LLDB TESTS ==================================================================================
// lldb-command:run

// lldb-command:print v
// lldbg-check:[...]$0 = size=3 { [0] = 1 [1] = 2 [2] = 3 }
// lldbr-check:(alloc::vec::Vec<i32>) v = size=3 { [0] = 1 [1] = 2 [2] = 3 }
// lldb-command:print zs
// lldbg-check:[...]$1 = { x = y = 123 z = w = 456 }
// lldbr-check:(issue_22656::StructWithZeroSizedField) zs = { x = y = 123 z = w = 456 }
// lldbr-command:continue

#![allow(unused_variables)]
#![allow(dead_code)]
#![feature(omit_gdb_pretty_printer_section)]
#![omit_gdb_pretty_printer_section]

struct ZeroSizedStruct;

struct StructWithZeroSizedField {
    x: ZeroSizedStruct,
    y: u32,
    z: ZeroSizedStruct,
    w: u64
}

fn main() {
    let v = vec![1,2,3];

    let zs = StructWithZeroSizedField {
        x: ZeroSizedStruct,
        y: 123,
        z: ZeroSizedStruct,
        w: 456
    };

    zzz(); // #break
}

fn zzz() { () }
