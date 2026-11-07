//@ run-pass
#![allow(dead_code)]
struct S<T, U = u16> {
    a: T,
    b: U,
}

trait Tr {
    type A;
}
impl Tr for u8 {
    type A = S<u8, u16>;
}

fn f<T: Tr<A = S<u8>>>() {
    let s = T::A { a: 0, b: 1 };
    match s {
        T::A { a, b } => {
            assert_eq!(a, 0);
            assert_eq!(b, 1);
        }
    }
}

fn main() {
    f::<u8>();
}
