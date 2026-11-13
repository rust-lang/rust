//@ revisions: msvc non-msvc

// This test makes sure that the LLDB pretty printer does not throw an exception
// when trying to handle a Vec<> or anything else that contains zero-sized
// fields.

//@ [msvc] only-msvc
//@ [non-msvc] ignore-msvc
//@ ignore-gdb

//@ compile-flags:-g
//@ disable-gdb-pretty-printers

// === LLDB TESTS ==================================================================================
//@ lldb-command:run

//@ lldb-command:v v
//@ lldb-check:[...] size=3 { [0] = 1 [1] = 2 [2] = 3 }
//@ lldb-command:v zs
//@ [non-msvc] lldb-check:[...] {x:{}, y:123, z:{}, w:456}
// LLDB's PDB handling implictly ignores ZSTs
//@ [msvc] lldb-check:[...] {y:123, w:456}

#![allow(unused_variables)]
#![allow(dead_code)]

struct ZeroSizedStruct;

#[repr(C)]
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
