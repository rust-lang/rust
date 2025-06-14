//@ build-fail

#![feature(link_llvm_intrinsics, abi_unadjusted)]

extern "unadjusted" {
    #[link_name = "llvm.abcde"]
    fn foo();
    //~^ ERROR: Invalid LLVM Intrinsic `llvm.abcde`
}

pub fn main() {
    unsafe { foo() }
}
