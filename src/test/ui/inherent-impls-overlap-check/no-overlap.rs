// run-pass
// aux-build:repeat.rs

// This tests the allocating algo branch of the
// inherent impls overlap checker.
// This branch was added by PR:
// https://github.com/rust-lang/rust/pull/78317
// In this test, we repeat many impl blocks
// to trigger the allocating branch.

#![allow(unused)]

extern crate repeat;

// Simple case where each impl block is distinct

struct Foo {}

repeat::repeat_with_idents!(impl Foo { fn IDENT() {} });

// There are overlapping impl blocks but due to generics,
// they may overlap.

struct Bar<T>(T);

struct A;
struct B;

repeat::repeat_with_idents!(impl Bar<A> { fn IDENT() {} });

impl Bar<A> { fn foo() {} }
impl Bar<B> { fn foo() {} }

fn main() {}
