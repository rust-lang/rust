//@ compile-flags: -Copt-level=3

#![crate_type = "lib"]

use std::mem::MaybeUninit;

// We need to make sure len is at offset 0, otherwise codegen needs an extra instruction
#[repr(C)]
pub struct SmallVec<T> {
    pub len: u64,
    pub arr: [MaybeUninit<T>; 24],
}

// CHECK-LABEL: @uninit_arr_via_const
#[no_mangle]
pub fn uninit_arr_via_const() -> SmallVec<String> {
    // CHECK-NEXT: start:
    // CHECK-NEXT: store i64 0,
    // CHECK-NEXT: ret
    SmallVec { len: 0, arr: [const { MaybeUninit::uninit() }; 24] }
}
