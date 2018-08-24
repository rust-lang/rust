#![feature(const_fn)]
const fn f() -> usize {
    5
}
struct A {
    field: usize,
}
fn main() {
    let _ = [0; f()];
}
