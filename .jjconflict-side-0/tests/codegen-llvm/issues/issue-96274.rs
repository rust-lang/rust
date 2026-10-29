//@ compile-flags: -Copt-level=3

#![crate_type = "lib"]

use std::mem::MaybeUninit;

pub fn maybe_uninit() -> [MaybeUninit<u8>; 3000] {
    // CHECK-NOT: memset
    [MaybeUninit::uninit(); 3000]
}

pub fn maybe_uninit_const<T>() -> [MaybeUninit<T>; 8192] {
    // CHECK-NOT: memset
    [const { MaybeUninit::uninit() }; 8192]
}
