// edition:2021
// revisions: current next
//[next] compile-flags: -Ztrait-solver=next
// check-pass

#![feature(type_alias_impl_trait)]

struct Foo;

impl Trait for Foo {}
pub trait Trait {}

pub type TAIT<T> = impl Trait;

async fn foo<T>() -> TAIT<T> {
    Foo
}

fn main() {}
