//@ revisions: no_flag with_flag
//@ [no_flag] check-pass
//@ [with_flag] compile-flags: -Zextra-const-ub-checks
#![feature(never_type)]
#![allow(unnecessary_transmutes)]

use std::mem::transmute;
use std::ptr::addr_of;

#[derive(Clone, Copy)]
enum E { A, B }

#[derive(Clone, Copy)]
enum Never {}

#[repr(usize)]
enum PtrSizedEnum { V }

// An enum with uninhabited variants but also at least 2 inhabited variants -- so the uninhabited
// variants *do* have a discriminant.
#[derive(Clone, Copy)]
enum UninhDiscriminant {
    A,
    B(!),
    C,
    D(Never),
}

const INVALID_BOOL: () = unsafe {
    let _x: bool = transmute(3u8);
    //[with_flag]~^ ERROR: evaluation of constant value failed
    //[with_flag]~| NOTE invalid value
};

const INVALID_PTR_IN_INT: () = unsafe {
    let _x: usize = transmute(&3u8);
    //[with_flag]~^ ERROR: evaluation of constant value failed
    //[with_flag]~| NOTE invalid value
};

const INVALID_PTR_IN_ENUM: () = unsafe {
    let _x: PtrSizedEnum = transmute(&3u8);
    //[with_flag]~^ ERROR: evaluation of constant value failed
    //[with_flag]~| NOTE invalid value
};

const INVALID_SLICE_TO_USIZE_TRANSMUTE: () = unsafe {
    let x: &[u8] = &[0; 32];
    let _x: (usize, usize) = transmute(x);
    //[with_flag]~^ ERROR: evaluation of constant value failed
    //[with_flag]~| NOTE invalid value
};

const UNALIGNED_PTR: () = unsafe {
    let _x: &u32 = transmute(&[0u8; 4]);
    //[with_flag]~^ ERROR: evaluation of constant value failed
    //[with_flag]~| NOTE invalid value
};

const UNINHABITED_VARIANT: () = unsafe {
    let data = [1u8];
    // Not using transmute, we want to hit the ImmTy code path.
    let v = *addr_of!(data).cast::<UninhDiscriminant>();
    //[with_flag]~^ ERROR: evaluation of constant value failed
    //[with_flag]~| NOTE invalid value
};

const PARTIAL_POINTER: () = unsafe {
    #[repr(C, packed)]
    struct Packed {
        pad1: u8,
        ptr: *const u8,
        pad2: [u8; 7],
    }
    // `Align` ensures that the entire thing has pointer alignment again.
    #[repr(C)]
    struct Align {
        p: Packed,
        align: usize,
    }
    let mem = Packed { pad1: 0, ptr: &0u8 as *const u8, pad2: [0; 7] };
    let mem = Align { p: mem, align: 0 };
    let _val = *(&mem as *const Align as *const [*const u8; 2]);
    //[with_flag]~^ ERROR: evaluation of constant value failed
    //[with_flag]~| NOTE invalid value
};

// Regression tests for an ICE (related to <https://github.com/rust-lang/rust/issues/113988>).
const VALID_ENUM1: E = { let e = E::A; e };
const VALID_ENUM2: Result<&'static [u8], ()> = { let e = Err(()); e };

// Htting the (non-integer) array code in validation with an immediate local.
const VALID_ARRAY: [Option<i32>; 0] = { let e = [None; 0]; e };

// Detecting oversized references.
const OVERSIZED_REF: () = { unsafe {
    let slice: *const [u8] = transmute((1usize, usize::MAX));
    let _val = &*slice;
    //[with_flag]~^ ERROR: evaluation of constant value failed
    //[with_flag]~| NOTE slice is bigger than largest supported object
} };

fn main() {}
