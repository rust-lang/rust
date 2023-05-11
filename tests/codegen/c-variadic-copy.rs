// Tests that `VaListImpl::clone` gets inlined into a call to `llvm.va_copy`

#![crate_type = "lib"]
#![feature(c_variadic)]
#![no_std]
use core::ffi::VaList;

extern "C" {
    fn foreign_c_variadic_1(_: VaList, ...);
}

pub unsafe extern "C" fn clone_variadic(ap: VaList) {
    let mut ap2 = ap.clone();
    // CHECK: call void @llvm.va_copy
    foreign_c_variadic_1(ap2.as_va_list(), 42i32);
}
