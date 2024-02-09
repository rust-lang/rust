// edition:2018
// check-pass

#![feature(async_closure, type_alias_impl_trait)]

type Tait = impl async Fn();
fn tait() -> Tait {
    || async {}
}

fn foo(x: impl async Fn()) -> impl async Fn() { x }

fn param<T: async Fn()>() {}

fn main() {}
