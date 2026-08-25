//@ edition:2021
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@ check-pass

#![feature(type_alias_impl_trait)]

struct Foo;

impl Trait for Foo {}
pub trait Trait {}

pub type TAIT<T> = impl Trait;

#[define_opaque(TAIT)]
async fn foo<T>() -> TAIT<T> {
    Foo
}

fn main() {}
