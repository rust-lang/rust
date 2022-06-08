// build-pass (FIXME(62277): could be check-pass?)
// edition:2018

#![feature(return_position_impl_trait_v2)]

use std::future::Future;

pub trait Foo {
    fn foo(&self) {}
}

pub fn call_with_ref_block<'a>(f: &'a (impl Foo + 'a)) -> impl Future<Output = ()> + 'a {
    async move { f.foo() }
}

fn main() {}
