//@ add-minicore
//@ compile-flags: -Copt-level=3
#![feature(c_variadic)]
#![crate_type = "lib"]

unsafe extern "C" {
    fn g(v: *mut u8);
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn f(mut args: ...) {
    // CHECK: call void @llvm.va_start
    unsafe { g(&raw mut args as *mut u8) }
    // We expect one call to the LLVM va_end from our desugaring of `...`. The `Drop` implementation
    // should only call the rust va_end intrinsic, which is a no-op.
    //
    // CHECK: call void @llvm.va_end
    // CHECK-NOT: call void @llvm.va_end
}
