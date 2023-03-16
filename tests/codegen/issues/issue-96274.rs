// min-llvm-version: 15.0
// compile-flags: -O

#![crate_type = "lib"]
#![feature(inline_const)]

use std::mem::MaybeUninit;

pub fn maybe_uninit() -> [MaybeUninit<u8>; 3000] {
    // CHECK-NOT: memset
    [MaybeUninit::uninit(); 3000]
}

pub fn maybe_uninit_const<T>() -> [MaybeUninit<T>; 8192] {
    // CHECK-NOT: memset
    [const { MaybeUninit::uninit() }; 8192]
}
