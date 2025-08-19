// This test makes sure that the LLDB pretty printer does not throw an exception
// when trying to handle a Vec<> or anything else that contains zero-sized
// fields.

//@ ignore-gdb

//@ compile-flags:-g
//@ disable-gdb-pretty-printers

// === LLDB TESTS ==================================================================================
// lldb-command:run

// lldb-command:v v
// lldb-check:[...] size=3 { [0] = 1 [1] = 2 [2] = 3 }
// lldb-command:v zs
// lldb-check:[...] { x = y = 123 z = w = 456 }

#![allow(unused_variables)]
#![allow(dead_code)]

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
