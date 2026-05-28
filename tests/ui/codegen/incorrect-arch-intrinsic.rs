//@ build-fail
//@ ignore-s390x
//@ normalize-stderr: "target arch `(.*)`" -> "target arch `TARGET_ARCH`"
//@ ignore-backends: gcc

#![feature(link_llvm_intrinsics, abi_unadjusted)]

extern "unadjusted" {
    #[link_name = "llvm.s390.sfpc"]
    fn foo(a: i32);
    //~^ ERROR: intrinsic `llvm.s390.sfpc` cannot be used with target arch
}

pub fn main() {
    unsafe {
        foo(0);
    }
}
