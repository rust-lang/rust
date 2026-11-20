//@ build-fail
//@ revisions: normal deprecated
//@ [normal] ignore-s390x
//@ [deprecated] ignore-aarch64
//@ normalize-stderr: "target arch `(.*)`" -> "target arch `TARGET_ARCH`"
//@ ignore-backends: gcc

#![feature(link_llvm_intrinsics)]

#[cfg(normal)]
extern "llvm-intrinsic" {
    #[link_name = "llvm.s390.sfpc"]
    fn foo(a: i32);
    //[normal]~^ ERROR: intrinsic `llvm.s390.sfpc` cannot be used with target arch
}

#[cfg(deprecated)]
extern "llvm-intrinsic" {
    #[link_name = "llvm.aarch64.neon.rbit.v8i8"]
    fn foo(a: i32);
    //[deprecated]~^ ERROR: intrinsic `llvm.aarch64.neon.rbit.v8i8` cannot be used with target arch
}

pub fn main() {
    unsafe {
        foo(0);
    }
}
