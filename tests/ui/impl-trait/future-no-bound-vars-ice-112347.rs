// issue: rust-lang/rust#112347
// ICE future has no bound vars
//@ edition:2021
//@ check-pass

#![feature(type_alias_impl_trait)]

use std::future::Future;

pub type Fut<'a> = impl Future<Output = ()> + 'a;

#[define_opaque(Fut)]
fn foo<'a>(_: &()) -> Fut<'_> {
    async {}
}

trait Test {
    fn hello();
}

impl Test for ()
where
    for<'a> Fut<'a>: Future<Output = ()>,
{
    fn hello() {}
}

fn main() {
    <()>::hello();
}
