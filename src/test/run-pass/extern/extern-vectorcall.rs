// run-pass
// ignore-arm
// ignore-aarch64

#![feature(abi_vectorcall)]

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
}
