// error-pattern: any use of this value will cause an error

#![feature(never_type)]
#![feature(const_maybe_uninit_assume_init, const_assert_type2)]
#![feature(core_intrinsics)]

use std::intrinsics;

#[allow(invalid_value)]
fn main() {
    use std::mem::MaybeUninit;

    const _BAD1: () = unsafe {
        MaybeUninit::<!>::uninit().assume_init();
    };
    const _BAD2: () = unsafe {
        intrinsics::assert_uninit_valid::<bool>();
    };
    const _BAD3: () = unsafe {
        intrinsics::assert_zero_valid::<&'static i32>();
    };
}
