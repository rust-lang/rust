//@ build-fail
//@ ignore-backends: gcc

#![feature(link_llvm_intrinsics)]

extern "llvm-intrinsic" {
    #[link_name = "llvm.abcde"]
    fn foo();
    //~^ ERROR: unknown LLVM intrinsic `llvm.abcde`
}

pub fn main() {
    unsafe { foo() }
}
