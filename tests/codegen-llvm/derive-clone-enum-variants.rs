//! Regression test for bad codegen of `#[derive(Clone)]` on enums.
//! Ensures efficient LLVM IR without unnecessary branches.
//! See <https://github.com/rust-lang/rust/issues/69174>.

//@ compile-flags: -C opt-level=3

#![crate_type = "lib"]

#[derive(Clone)]
pub enum Foo {
    A(u8),
    B(bool),
}

#[derive(Clone)]
pub enum Bar {
    C(Foo),
    D(u8),
}

// CHECK-LABEL: @clone_bar
// CHECK-NOT: icmp
#[unsafe(no_mangle)]
pub fn clone_bar(b: &Bar) -> Bar {
    b.clone()
}
