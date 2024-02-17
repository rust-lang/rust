//@ run-pass
#![allow(dead_code)]
const fn f() -> usize {
    5
}
struct A {
    field: usize,
}
fn main() {
    let _ = [0; f()];
}
