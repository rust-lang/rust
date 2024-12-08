//@ run-pass
#![allow(dead_code)]

#[derive(Clone)]
enum E<T,U> {
    A(T),
    B(T,U),
    C
}

pub fn main() {
    let _ = E::A::<isize, isize>(1).clone();
}
