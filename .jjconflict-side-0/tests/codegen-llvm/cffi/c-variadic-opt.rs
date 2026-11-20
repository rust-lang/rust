//@ compile-flags: -C opt-level=3
//@ ignore-bpf: BPF does not support C variadics

#![crate_type = "lib"]
#![no_std]
use core::ffi::VaList;

extern "C" {
    fn vprintf(fmt: *const i8, ap: VaList) -> i32;
}

// Ensure that `va_start` and `va_end` are properly injected even
// when the "spoofed" `VaList` is not used.
#[no_mangle]
pub unsafe extern "C" fn c_variadic_no_use(fmt: *const i8, mut ap: ...) -> i32 {
    // CHECK: call void @llvm.va_start
    vprintf(fmt, ap)
    // We no longer call the LLVM va_end.
    // CHECK-NOT: call void @llvm.va_end
}
