// Ensures wide pointers to dynamically sized types are visualized properly. Note that this does
// not include slice/str wide pointers, only wide pointers that point to dynamically sized
// user defined types.

//@ compile-flags:-g
//@ min-llvm-lldb-version: 21.1.0
//@ min-apple-lldb-version: 1703.0.236.21

// LLDB completely omits ZST-typed fields from the parsed *type*:
// ```
// ignore-tidy-linelength
// (lldb) script lldb.frame.var("a").GetNonSyntheticValue().GetChildMemberWithName("data_ptr").Dereference().GetType()
// struct A {
//     unsigned long long x;
//     unsigned short y;
// }
// ```
//
// Because of this, we can't even attempt to make this visualizer work on MSVC because we have no
// way to know that there's an unsized tail *at all*, let alone what type it's supposed to be.

//@ ignore-msvc

//@ ignore-backends: gcc


// === LLDB TESTS ==================================================================================

//@ lldb-command:run

// prevents `unsigned char[]` for printing as a string.
//@ lldb-command:type category disable system

//@ lldb-repr:a
//@ lldb-repr:b
//@ lldb-repr:a_ref
//@ lldb-repr:b_ref


#![allow(unused)]
use std::{
    alloc::{Layout, alloc_zeroed},
    ptr,
};

// `Sized` and `Unsized` as well as `A` and `B` are laid out in such a way that if we stop
// outputting the correct debug info for `[T]`, they will be impossible to tell apart from their
// layouts in the DWARF/PDB data. Not only are the types all the same size, the unsized portions
// also "hide" within the natural padding of the parent struct

#[derive(Clone, Copy)]
#[repr(C)]
struct Sized {
    f1: u16,
    f2: u8,
    f3: u8,
}

#[repr(C)]
struct Unsized {
    f1: u16,
    f2: u8,
    f3: [u8],
}

#[repr(C)]
struct A {
    x: u64,
    y: u16,
    data: [Sized],
}

#[repr(C)]
struct B {
    x: u64,
    y: u16,
    data: Unsized,
}

fn main() -> Result<(), std::alloc::LayoutError> {
    unsafe {
        // 10 bytes
        let base_l = Layout::new::<u64>().extend(Layout::new::<u16>())?.0;

        // 10 + (4 * 3) = 22 bytes
        let a_l = base_l.extend(Layout::array::<Sized>(3)?)?.0;

        // 10 + 3 + (1 * 9) = 22 bytes
        let b_l = base_l
            .extend(Layout::new::<u16>())?
            .0
            .extend(Layout::new::<u8>())?
            .0
            .extend(Layout::array::<u8>(9)?)?
            .0;

        let a = ptr::slice_from_raw_parts(alloc_zeroed(a_l), 3) as *mut A;
        let b = ptr::slice_from_raw_parts_mut(alloc_zeroed(b_l), 9) as *mut B;

        let a_ref = a.as_mut().unwrap();

        a_ref.x = 1;
        a_ref.y = 2;
        a_ref.data.copy_from_slice(&[
            Sized {
                f1: 3,
                f2: 4,
                f3: 5,
            },
            Sized {
                f1: 6,
                f2: 7,
                f3: 8,
            },
            Sized {
                f1: 9,
                f2: 10,
                f3: 11,
            },
        ]);

        let b_ref = b.as_mut().unwrap();

        b_ref.x = 1;
        b_ref.y = 2;
        b_ref.data.f1 = 3;
        b_ref.data.f2 = 4;
        b_ref.data.f3.copy_from_slice(&[5, 6, 7, 8, 9, 10, 11, 12, 13]);

        Ok(()) // #break
    }
}
