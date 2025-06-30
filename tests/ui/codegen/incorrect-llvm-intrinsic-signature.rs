//@ build-fail

#![feature(link_llvm_intrinsics, abi_unadjusted)]

extern "unadjusted" {
    #[link_name = "llvm.assume"]
    fn foo();
    //~^ ERROR: Intrinsic signature mismatch for `llvm.assume`: expected signature `void (i1)`, found `void ()`
}

pub fn main() {
    unsafe { foo() }
}
