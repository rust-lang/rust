//@ add-minicore
//@ compile-flags: -Copt-level=3
#![crate_type = "lib"]

unsafe extern "C" {
    fn g(v: *mut u8);
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn f(mut args: ...) {
    // CHECK: call void @llvm.va_start
    unsafe { g(&raw mut args as *mut u8) }
    // We no longer call the LLVM va_end.
    // CHECK-NOT: call void @llvm.va_end
}
