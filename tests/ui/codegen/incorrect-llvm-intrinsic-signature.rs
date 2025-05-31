//@ build-fail

#![feature(link_llvm_intrinsics, abi_unadjusted)]
#![allow(internal_features, non_camel_case_types, improper_ctypes)]

extern "unadjusted" {
    #[link_name = "llvm.assume"]
    fn foo();
}

pub fn main() {
    unsafe { foo() }
}

//~? ERROR: Intrinsic signature mismatch for `llvm.assume`: expected signature `void (i1)`
