// https://github.com/rust-lang/rust/issues/43754

#![feature(const_fn)]
const fn foo(x: usize) -> usize {
    return x;
}
fn main() {
    [0; foo(2)];
}
