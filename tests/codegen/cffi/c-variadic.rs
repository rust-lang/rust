//@ needs-unwind
//@ compile-flags: -C no-prepopulate-passes -Copt-level=0
//

#![crate_type = "lib"]
#![feature(c_variadic)]
#![no_std]
use core::ffi::VaList;

extern "C" {
    fn foreign_c_variadic_0(_: i32, ...);
    fn foreign_c_variadic_1(_: VaList, ...);
}

pub unsafe extern "C" fn use_foreign_c_variadic_0() {
    // Ensure that we correctly call foreign C-variadic functions.
    // CHECK: call void (i32, ...) @foreign_c_variadic_0([[PARAM:i32( signext)?]] 0)
    foreign_c_variadic_0(0);
    // CHECK: call void (i32, ...) @foreign_c_variadic_0([[PARAM]] 0, [[PARAM]] 42)
    foreign_c_variadic_0(0, 42i32);
    // CHECK: call void (i32, ...) @foreign_c_variadic_0([[PARAM]] 0, [[PARAM]] 42, [[PARAM]] 1024)
    foreign_c_variadic_0(0, 42i32, 1024i32);
    // CHECK: call void (i32, ...) @foreign_c_variadic_0([[PARAM]] 0, [[PARAM]] 42, [[PARAM]] 1024, [[PARAM]] 0)
    foreign_c_variadic_0(0, 42i32, 1024i32, 0i32);
}

// Ensure that we do not remove the `va_list` passed to the foreign function when
// removing the "spoofed" `VaListImpl` that is used by Rust defined C-variadics.
pub unsafe extern "C" fn use_foreign_c_variadic_1_0(ap: VaList) {
    // CHECK: call void ({{.*}}, ...) @foreign_c_variadic_1({{.*}} %ap)
    foreign_c_variadic_1(ap);
}

pub unsafe extern "C" fn use_foreign_c_variadic_1_1(ap: VaList) {
    // CHECK: call void ({{.*}}, ...) @foreign_c_variadic_1({{.*}} %ap, [[PARAM]] 42)
    foreign_c_variadic_1(ap, 42i32);
}
pub unsafe extern "C" fn use_foreign_c_variadic_1_2(ap: VaList) {
    // CHECK: call void ({{.*}}, ...) @foreign_c_variadic_1({{.*}} %ap, [[PARAM]] 2, [[PARAM]] 42)
    foreign_c_variadic_1(ap, 2i32, 42i32);
}

pub unsafe extern "C" fn use_foreign_c_variadic_1_3(ap: VaList) {
    // CHECK: call void ({{.*}}, ...) @foreign_c_variadic_1({{.*}} %ap, [[PARAM]] 2, [[PARAM]] 42, [[PARAM]] 0)
    foreign_c_variadic_1(ap, 2i32, 42i32, 0i32);
}

// Ensure that `va_start` and `va_end` are properly injected.
#[no_mangle]
pub unsafe extern "C" fn c_variadic(n: i32, mut ap: ...) -> i32 {
    // CHECK: call void @llvm.va_start
    let mut sum = 0;
    for _ in 0..n {
        sum += ap.arg::<i32>();
    }
    sum
    // CHECK: call void @llvm.va_end
}

// Ensure that we generate the correct `call` signature when calling a Rust
// defined C-variadic.
pub unsafe fn test_c_variadic_call() {
    // CHECK: call [[RET:(signext )?i32]] (i32, ...) @c_variadic([[PARAM]] 0)
    c_variadic(0);
    // CHECK: call [[RET]] (i32, ...) @c_variadic([[PARAM]] 0, [[PARAM]] 42)
    c_variadic(0, 42i32);
    // CHECK: call [[RET]] (i32, ...) @c_variadic([[PARAM]] 0, [[PARAM]] 42, [[PARAM]] 1024)
    c_variadic(0, 42i32, 1024i32);
    // CHECK: call [[RET]] (i32, ...) @c_variadic([[PARAM]] 0, [[PARAM]] 42, [[PARAM]] 1024, [[PARAM]] 0)
    c_variadic(0, 42i32, 1024i32, 0i32);
}
