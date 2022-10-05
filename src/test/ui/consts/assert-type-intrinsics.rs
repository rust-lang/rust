// error-pattern: any use of this value will cause an error

#![feature(never_type)]
#![feature(const_assert_type2)]
#![feature(core_intrinsics)]

use std::intrinsics;

#[allow(invalid_value)]
fn main() {
    use std::mem::MaybeUninit;

    const _BAD1: () = unsafe {
        intrinsics::assert_inhabited::<!>(); //~ERROR: any use of this value will cause an error
        //~^WARN: previously accepted
    };
    const _BAD2: () = {
        intrinsics::assert_uninit_valid::<!>(); //~ERROR: any use of this value will cause an error
        //~^WARN: previously accepted
    };
    const _BAD3: () = {
        intrinsics::assert_zero_valid::<&'static i32>(); //~ERROR: any use of this value will cause an error
        //~^WARN: previously accepted
    };
}
