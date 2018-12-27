// run-pass
use std::ops::Add;

struct S<T, U = u16> {
    a: T,
    b: U,
}

trait Tr {
    fn f(&self) -> Self;
}

impl<T: Default + Add<u8, Output = T>, U: Default> Tr for S<T, U> {
    fn f(&self) -> Self {
        let s = Self { a: Default::default(), b: Default::default() };
        match s {
            Self { a, b } => Self { a: a + 1, b: b }
        }
    }
}

impl<T: Default, U: Default + Add<u16, Output = U>> S<T, U> {
    fn g(&self) -> Self {
        let s = Self { a: Default::default(), b: Default::default() };
        match s {
            Self { a, b } => Self { a: a, b: b + 1 }
        }
    }
}

impl S<u8> {
    fn new() -> Self {
        Self { a: 0, b: 1 }
    }
}

fn main() {
    let s0 = S::new();
    let s1 = s0.f();
    assert_eq!(s1.a, 1);
    assert_eq!(s1.b, 0);
    let s2 = s0.g();
    assert_eq!(s2.a, 0);
    assert_eq!(s2.b, 1);
}
