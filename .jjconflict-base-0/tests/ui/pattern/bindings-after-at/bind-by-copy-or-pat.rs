//@ run-pass
#![allow(unused)]

struct A(u32);

pub fn main() {
    // Bindings are lowered in the order they appear syntactically, so this works.
    let x @ (A(a) | A(a)) = A(10);
    assert!(x.0 == 10);
    assert!(a == 10);

    // This also works.
    let (x @ A(a) | x @ A(a)) = A(10);
    assert!(x.0 == 10);
    assert!(a == 10);
}
