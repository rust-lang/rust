//! Regression test for https://github.com/rust-lang/rust/issues/104037.
//! LLVM used to hit an assertion "Vector elements must have same size"
//! when compiling derived Clone with MIR optimisation level of 3.

//@ build-pass
//@ compile-flags: -Zmir-opt-level=3 -Copt-level=3

#[derive(Clone)]
pub struct Foo(Bar, u32);

#[derive(Clone, Copy)]
pub struct Bar(u8, u8, u8);

fn main() {
    let foo: Vec<Foo> = Vec::new();
    let _ = foo.clone();
}
