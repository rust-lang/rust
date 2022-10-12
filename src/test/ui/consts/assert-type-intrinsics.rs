#![feature(never_type)]
#![feature(const_assert_type2)]
#![feature(core_intrinsics)]

use std::intrinsics;

#[allow(invalid_value)]
fn main() {
    use std::mem::MaybeUninit;

    const _BAD1: () = unsafe {
        MaybeUninit::<!>::uninit().assume_init();
        //~^ERROR: evaluation of constant value failed
    };
    const _BAD2: () = {
        intrinsics::assert_uninit_valid::<&'static i32>();
        //~^ERROR: evaluation of constant value failed
    };
    const _BAD3: () = {
        intrinsics::assert_zero_valid::<&'static i32>();
        //~^ERROR: evaluation of constant value failed
    };
}
