//@ build-fail
//@ ignore-backends: gcc

#![feature(link_llvm_intrinsics, abi_unadjusted)]

extern "unadjusted" {
    #[link_name = "llvm.abcde"]
    fn foo();
    //~^ ERROR: unknown LLVM intrinsic `llvm.abcde`
}

pub fn main() {
    unsafe { foo() }
}
