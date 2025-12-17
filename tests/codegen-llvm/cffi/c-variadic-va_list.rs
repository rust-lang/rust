//@ needs-unwind
//@ compile-flags: -Copt-level=3
//@ min-llvm-version: 21

#![crate_type = "lib"]
#![feature(c_variadic)]
#![no_std]
use core::ffi::VaList;

// Ensure that we do not remove the `va_list` passed to the foreign function when
// removing the "spoofed" `VaList` that is used by Rust defined C-variadics.

extern "C" {
    fn foreign_c_variadic_1(_: VaList, ...);
}

// CHECK-LABEL: use_foreign_c_variadic_1_0
pub unsafe extern "C" fn use_foreign_c_variadic_1_0(ap: VaList) {
    // CHECK: call void ({{.*}}, ...) @foreign_c_variadic_1({{.*}} %ap)
    foreign_c_variadic_1(ap);
}

// CHECK-LABEL: use_foreign_c_variadic_1_1
pub unsafe extern "C" fn use_foreign_c_variadic_1_1(ap: VaList) {
    // CHECK: call void ({{.*}}, ...) @foreign_c_variadic_1({{.*}} %ap, i32 noundef{{( signext)?}} 42)
    foreign_c_variadic_1(ap, 42i32);
}
pub unsafe extern "C" fn use_foreign_c_variadic_1_2(ap: VaList) {
    // CHECK: call void ({{.*}}, ...) @foreign_c_variadic_1({{.*}} %ap, i32 noundef{{( signext)?}} 2, i32 noundef{{( signext)?}} 42)
    foreign_c_variadic_1(ap, 2i32, 42i32);
}

pub unsafe extern "C" fn use_foreign_c_variadic_1_3(ap: VaList) {
    // CHECK: call void ({{.*}}, ...) @foreign_c_variadic_1({{.*}} %ap, i32 noundef{{( signext)?}} 2, i32 noundef{{( signext)?}} 42, i32 noundef{{( signext)?}} 0)
    foreign_c_variadic_1(ap, 2i32, 42i32, 0i32);
}
