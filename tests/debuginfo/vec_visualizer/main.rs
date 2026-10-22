#![allow(unused)]
// Ensures `Vec<T>` visualizers still work as expected

//@ compile-flags:-g
//@ min-llvm-lldb-version: 21.1.0
//@ min-apple-lldb-version: 1703.0.236.21

//@ ignore-backends: gcc

// === LLDB TESTS ==================================================================================
//@ lldb-command:run
//@ lldb-repr:vec_primitive
//@ lldb-repr:vec_ref
//@ lldb-repr:vec_endpad
//@ lldb-repr:vec_midpad
//@ lldb-repr:vec_generic_i32
//@ lldb-repr:vec_generic_udt

// #FIXME(Walnut356): The visualizers don't really handle these correctly. They don't break exactly,
// but the output is awkward in a way that `debugger_tester` can't handle (i.e. it reports having
// children, but those children are invalid because LLDB doesn't like interacting with ZSTs).
// //@ lldb-repr:vec_zst

//@ lldb-repr:ref_vec
//@ lldb-repr:ptr_vec
//@ lldb-repr:ref_mut_vec
//@ lldb-repr:ptr_mut_vec
//@ lldb-repr:some_vec
//@ lldb-repr:some_empty_vec

use std::mem::offset_of;

#[repr(C)]
struct Point<T> {
    x: T,
    y: T,
}

// The following 2 structs ensure the debugger can properly locate sequential padded values
#[repr(C)]
struct EndPad {
    a: u32,
    b: u8,
}

// u32 + u8 = 5 bytes total. If the struct is 8 bytes and a 1-byte value is last, we know there
// are 3 pad bytes at the end
const _: () = assert!(size_of::<EndPad>() == 8 && offset_of!(EndPad, b) == 4);

#[repr(C)]
struct MidPad {
    a: u8,
    b: u32,
}

// u8 = 1 byte, so if the next field is at offset 4, we know we have 3 pad bytes in the middle
const _: () = assert!(offset_of!(MidPad, b) == 4);

struct Zst();

fn main() {
    // basic functionality
    let mut vec_primitive = vec![i32::MIN, -1, 0, 1, i32::MAX];
    let vec_ref = vec![&u32::MIN, &1, &2, &u32::MAX];
    let vec_endpad =
        vec![EndPad { a: 10, b: 20 }, EndPad { a: 30, b: 40 }, EndPad { a: 50, b: 60 }];
    let vec_midpad =
        vec![MidPad { a: 10, b: 20 }, MidPad { a: 30, b: 40 }, MidPad { a: 50, b: 60 }];

    // Make sure we can properly populate generic children, especially when a namespaced type is
    // used as a generic arg.
    let vec_generic_i32 = vec![Point { x: 0.0, y: 1.1 }, Point { x: 2.2, y: 3.3 }];
    let vec_generic_udt =
        vec![Point { x: vec![0, 0], y: vec![1, 1] }, Point { x: vec![2, 2], y: vec![3, 3] }];

    let vec_zst = vec![Zst(), Zst(), Zst(), Zst()];

    // wrappers
    let ref_vec = &vec_primitive;
    let ptr_vec = ref_vec as *const _;
    let ref_mut_vec = &mut vec_primitive;
    let ptr_mut_vec = ref_mut_vec as *mut _;
    let some_vec = Some(vec_primitive.clone());
    let some_empty_vec: Option<Vec<i32>> = Some(vec![]);
    // FIXME(min-lldb): re-enable once CI is updated to LLDB 23
    // let none_vec: Option<Vec<i32>> = None;

    zzz(); // #break
}

fn zzz() {}
