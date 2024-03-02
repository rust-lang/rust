//@ known-bug: unknown
#![allow(unused)]

struct A(u32);

pub fn main() {
    // The or-pattern bindings are lowered after `x`, which triggers the error.
    let x @ (A(a) | A(a)) = A(10);
    // ERROR: use of moved value
    assert!(x.0 == 10);
    assert!(a == 10);

    // This works.
    let (x @ A(a) | x @ A(a)) = A(10);
    assert!(x.0 == 10);
    assert!(a == 10);
}
