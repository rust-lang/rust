//@ edition:2018
//@ check-pass

#![feature(type_alias_impl_trait)]

type Tait = impl AsyncFn();
#[define_opaque(Tait)]
fn tait() -> Tait {
    || async {}
}

fn foo(x: impl AsyncFn()) -> impl AsyncFn() {
    x
}

fn param<T: AsyncFn()>() {}

fn main() {}
