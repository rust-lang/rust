//@ build-fail

#![feature(link_llvm_intrinsics, abi_unadjusted)]
#![allow(internal_features, non_camel_case_types, improper_ctypes)]

extern "unadjusted" {
    #[link_name = "llvm.abcde"]
    fn foo();
}

pub fn main() {
    unsafe { foo() }
}

//~? ERROR: Invalid LLVM intrinsic: `llvm.abcde`
