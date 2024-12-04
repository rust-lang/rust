//@ edition:2018
//@ check-pass

#![feature(async_closure, type_alias_impl_trait)]

type Tait = impl AsyncFn();
fn tait() -> Tait {
    || async {}
}

fn foo(x: impl AsyncFn()) -> impl AsyncFn() { x }

fn param<T: AsyncFn()>() {}

fn main() {}
