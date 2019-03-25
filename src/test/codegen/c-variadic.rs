// compile-flags: -C no-prepopulate-passes

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
    // CHECK: invoke void (i32, ...) @foreign_c_variadic_0(i32 0)
    foreign_c_variadic_0(0);
    // CHECK: invoke void (i32, ...) @foreign_c_variadic_0(i32 0, i32 42)
    foreign_c_variadic_0(0, 42i32);
    // CHECK: invoke void (i32, ...) @foreign_c_variadic_0(i32 0, i32 42, i32 1024)
    foreign_c_variadic_0(0, 42i32, 1024i32);
    // CHECK: invoke void (i32, ...) @foreign_c_variadic_0(i32 0, i32 42, i32 1024, i32 0)
    foreign_c_variadic_0(0, 42i32, 1024i32, 0i32);
}

// Ensure that we do not remove the `va_list` passed to the foreign function when
// removing the "spoofed" `VaListImpl` that is used by Rust defined C-variadics.
pub unsafe extern "C" fn use_foreign_c_variadic_1_0(ap: VaList) {
    // CHECK: invoke void ({{.*}}*, ...) @foreign_c_variadic_1({{.*}} %ap)
    foreign_c_variadic_1(ap);
}

pub unsafe extern "C" fn use_foreign_c_variadic_1_1(ap: VaList) {
    // CHECK: invoke void ({{.*}}*, ...) @foreign_c_variadic_1({{.*}} %ap, i32 42)
    foreign_c_variadic_1(ap, 42i32);
}
pub unsafe extern "C" fn use_foreign_c_variadic_1_2(ap: VaList) {
    // CHECK: invoke void ({{.*}}*, ...) @foreign_c_variadic_1({{.*}} %ap, i32 2, i32 42)
    foreign_c_variadic_1(ap, 2i32, 42i32);
}

pub unsafe extern "C" fn use_foreign_c_variadic_1_3(ap: VaList) {
    // CHECK: invoke void ({{.*}}*, ...) @foreign_c_variadic_1({{.*}} %ap, i32 2, i32 42, i32 0)
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
    // CHECK: call i32 (i32, ...) @c_variadic(i32 0)
    c_variadic(0);
    // CHECK: call i32 (i32, ...) @c_variadic(i32 0, i32 42)
    c_variadic(0, 42i32);
    // CHECK: call i32 (i32, ...) @c_variadic(i32 0, i32 42, i32 1024)
    c_variadic(0, 42i32, 1024i32);
    // CHECK: call i32 (i32, ...) @c_variadic(i32 0, i32 42, i32 1024, i32 0)
    c_variadic(0, 42i32, 1024i32, 0i32);
}
