//@ check-pass
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

//! This test checks that we can successfully infer
//! the hidden type of `FooImpl` to be `Foo<i32, {closure}>`
//! and `ImplT` to be `i32`. This test used to fail, because
//! we were unable to make the connection that the closure
//! argument is the same as the first argument of `Foo`.

#![feature(type_alias_impl_trait)]

use std::fmt::Debug;
use std::marker::PhantomData;

struct Foo<T: Debug, F: FnOnce(T)> {
    f: F,
    _phantom: PhantomData<T>,
}

type ImplT = impl Debug;
type FooImpl = Foo<ImplT, impl FnOnce(ImplT)>;

#[define_opaque(FooImpl)]
fn bar() -> FooImpl {
    Foo::<i32, _> { f: |_| (), _phantom: PhantomData }
}

fn main() {}
