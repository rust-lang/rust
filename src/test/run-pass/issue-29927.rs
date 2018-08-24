#![feature(const_fn)]
struct A {
    field: usize,
}
const fn f() -> usize {
    5
}
fn main() {
    let _ = [0; f()];
}
