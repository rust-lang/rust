//! Verify the arm64ec calling convention for `i128` passed through the variadic
//! portion of a C-variadic call.
//!
//! On arm64ec the variadic tail follows the MS x64 ABI: any argument that does not
//! fit in 8 bytes, or is not 1/2/4/8 bytes, is passed by reference. So a variadic
//! `i128`/`u128` is passed indirectly (as a pointer), which must stay in sync with
//! how `va_arg` reads it back. A *fixed* `i128` argument is still passed by value.

//@ add-minicore
//@ compile-flags: -Copt-level=3 --target arm64ec-pc-windows-msvc
//@ needs-llvm-components: aarch64

#![crate_type = "lib"]
#![no_std]
#![no_core]
#![feature(no_core)]

extern crate minicore;

extern "C" {
    fn variadic(fixed: u32, ...);
    fn fixed(arg: i128);
}

// A variadic `i128` argument is passed by reference (as a pointer).
#[no_mangle]
pub unsafe extern "C" fn pass_variadic_i128(x: i128) {
    // CHECK-LABEL: @pass_variadic_i128(
    // CHECK: call void (i32, ...) @variadic(i32 {{.*}}, ptr {{.*}})
    variadic(0, x);
}

// A fixed `i128` argument is still passed by value.
#[no_mangle]
pub unsafe extern "C" fn pass_fixed_i128(x: i128) {
    // CHECK-LABEL: @pass_fixed_i128(
    // CHECK: call void @fixed(i128
    fixed(x);
}
