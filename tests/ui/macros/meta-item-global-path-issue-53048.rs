// Regression test for https://github.com/rust-lang/rust/issues/53048/
// This test ensures that a global path in a meta item is accepted by the compiler.
// All examples are different ways of specifying the same path, and should all compile fine.
//@ check-pass

extern crate core;

#[derive(::core::clone::Clone)] pub struct A;
#[derive(core::clone::Clone)] pub struct B;
#[derive(Clone, ::core::marker::Copy)] pub struct C;
#[cfg_attr(true, derive(::core::clone::Clone))] pub struct G;

macro_rules! m { ($m:meta) => { #[derive($m)] pub struct S; }; }
m!(::core::clone::Clone);

fn main() {}
