// run-pass
// ignore-arm
// ignore-aarch64

trait A {
    extern "fastcall" fn test1(i: i32);
    extern fn test2(i: i32);
}

struct S;
impl S {
    extern "stdcall" fn test3(i: i32) {
        assert_eq!(i, 3);
    }
}

impl A for S {
    extern "fastcall" fn test1(i: i32) {
        assert_eq!(i, 1);
    }
    extern fn test2(i: i32) {
        assert_eq!(i, 2);
    }
}

fn main() {
    <S as A>::test1(1);
    <S as A>::test2(2);
    S::test3(3);
}
