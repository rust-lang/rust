// revisions: no_flag with_flag
// [no_flag] check-pass
// [with_flag] compile-flags: -Zextra-const-ub-checks
#![feature(never_type)]

use std::mem::transmute;
use std::ptr::addr_of;

#[derive(Clone, Copy)]
enum E { A, B }

#[derive(Clone, Copy)]
enum Never {}

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
    //[with_flag]~| invalid value
};

const INVALID_PTR_IN_INT: () = unsafe {
    let _x: usize = transmute(&3u8);
    //[with_flag]~^ ERROR: evaluation of constant value failed
};

const INVALID_SLICE_TO_USIZE_TRANSMUTE: () = unsafe {
    let x: &[u8] = &[0; 32];
    let _x: (usize, usize) = transmute(x);
    //[with_flag]~^ ERROR: evaluation of constant value failed
};

const UNALIGNED_PTR: () = unsafe {
    let _x: &u32 = transmute(&[0u8; 4]);
    //[with_flag]~^ ERROR: evaluation of constant value failed
    //[with_flag]~| invalid value
};

const UNINHABITED_VARIANT: () = unsafe {
    let data = [1u8];
    // Not using transmute, we want to hit the ImmTy code path.
    let v = *addr_of!(data).cast::<UninhDiscriminant>();
    //[with_flag]~^ ERROR: evaluation of constant value failed
};

// Regression tests for an ICE (related to <https://github.com/rust-lang/rust/issues/113988>).
const VALID_ENUM1: E = { let e = E::A; e };
const VALID_ENUM2: Result<&'static [u8], ()> = { let e = Err(()); e };

fn main() {}
