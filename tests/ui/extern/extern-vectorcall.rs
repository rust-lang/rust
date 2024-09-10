//@ run-pass
//@ aux-build:foreign-vectorcall.rs
//@ revisions: x64 x32
//@ [x64]only-x86_64
//@ [x32]only-x86

#![feature(abi_vectorcall)]

extern crate foreign_vectorcall;

// Import this as a foreign function, that's the code path we are interested in
// (LLVM has to do some name mangling here).
extern "vectorcall" {
    fn call_with_42(i: i32);
}

trait A {
    extern "vectorcall" fn test1(i: i32);
}

struct S;

impl A for S {
    extern "vectorcall" fn test1(i: i32) {
        assert_eq!(i, 1);
    }
}

extern "vectorcall" fn test2(i: i32) {
    assert_eq!(i, 2);
}

fn main() {
    <S as A>::test1(1);
    test2(2);
    unsafe { call_with_42(42) };
}
