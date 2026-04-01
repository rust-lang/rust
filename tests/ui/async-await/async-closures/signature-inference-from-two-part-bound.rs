//@ edition: 2021
//@ check-pass
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

use std::future::Future;
use std::any::Any;

struct Struct;
impl Struct {
    fn method(&self) {}
}

fn fake_async_closure<F, Fut>(_: F)
where
    F: Fn(Struct) -> Fut,
    Fut: Future<Output = ()>,
{}

fn main() {
    fake_async_closure(async |s| {
        s.method();
    })
}
