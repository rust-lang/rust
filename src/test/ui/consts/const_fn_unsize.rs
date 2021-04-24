// gate-test-const_fn_unsize

// revisions: stock gated

#![feature(rustc_attrs)]
#![cfg_attr(gated, feature(const_fn_unsize))]

use std::ptr::NonNull;

const fn test() {
    let _x = NonNull::<[i32; 0]>::dangling() as NonNull<[i32]>;
    //[stock]~^ unsizing cast
}

#[rustc_error]
fn main() {} //[gated]~ fatal error triggered by #[rustc_error]
