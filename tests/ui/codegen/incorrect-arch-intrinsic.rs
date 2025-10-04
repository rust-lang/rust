//@ build-fail
//@ ignore-s390x
//@ normalize-stderr: "target arch `(.*)`" -> "target arch `TARGET_ARCH`"

#![feature(link_llvm_intrinsics, abi_unadjusted)]

extern "unadjusted" {
    #[link_name = "llvm.s390.sfpc"]
    fn foo(a: i32);
    //~^ ERROR: Intrinsic `llvm.s390.sfpc` cannot be used with target arch
}

pub fn main() {
    unsafe {
        foo(0);
    }
}
