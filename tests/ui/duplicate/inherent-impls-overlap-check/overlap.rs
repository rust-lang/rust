//@ proc-macro: repeat.rs

#![allow(unused)]

// This tests the allocating algo branch of the
// inherent impls overlap checker.
// This branch was added by PR:
// https://github.com/rust-lang/rust/pull/78317
// In this test, we repeat many impl blocks
// to trigger the allocating branch.

// Simple overlap

extern crate repeat;

struct Foo {}

repeat::repeat_with_idents!(impl Foo { fn IDENT() {} });

impl Foo { fn hello() {} } //~ERROR duplicate definitions with name `hello`
impl Foo { fn hello() {} }

// Transitive overlap

struct Foo2 {}

repeat::repeat_with_idents!(impl Foo2 { fn IDENT() {} });

impl Foo2 {
    fn bar() {}
    fn hello2() {} //~ERROR duplicate definitions with name `hello2`
}

impl Foo2 {
    fn baz() {}
    fn hello2() {}
}

// Slightly stronger transitive overlap

struct Foo3 {}

repeat::repeat_with_idents!(impl Foo3 { fn IDENT() {} });

impl Foo3 {
    fn bar() {} //~ERROR duplicate definitions with name `bar`
    fn hello3() {} //~ERROR duplicate definitions with name `hello3`
}

impl Foo3 {
    fn bar() {}
    fn hello3() {}
}

// Generic overlap

struct Bar<T>(T);

struct A;
struct B;

repeat::repeat_with_idents!(impl Bar<A> { fn IDENT() {} });

impl Bar<A> { fn foo() {} fn bar2() {} }
impl Bar<B> {
    fn foo() {}
    fn bar2() {} //~ERROR duplicate definitions with name `bar2`
}
impl Bar<B> { fn bar2() {} }

fn main() {}
