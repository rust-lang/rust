// aux-build:derive-a.rs
// ignore-stage1

#[macro_use]
extern crate derive_a;

#[derive(Debug, PartialEq, A, Eq, Copy, Clone)]
struct A;

fn main() {
    A;
    assert_eq!(A, A);
    A.clone();
    let a = A;
    let _c = a;
    let _d = a;
}
