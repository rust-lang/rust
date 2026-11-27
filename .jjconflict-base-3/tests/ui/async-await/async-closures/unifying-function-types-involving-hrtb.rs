//! Regresssion test for <https://github.com/rust-lang/rust/issues/59337>.

//@ edition:2018
//@ check-pass

use std::future::Future;

trait Foo<'a> {
    type Future: Future<Output = u8> + 'a;

    fn start(self, f: &'a u8) -> Self::Future;
}

impl<'a, Fn, Fut> Foo<'a> for Fn
where
    Fn: FnOnce(&'a u8) -> Fut,
    Fut: Future<Output = u8> + 'a,
{
    type Future = Fut;

    fn start(self, f: &'a u8) -> Self::Future { (self)(f) }
}

fn foo<F>(f: F) where F: for<'a> Foo<'a> {
    let bar = 5;
    f.start(&bar);
}

fn main() {
    foo(async move | f: &u8 | { *f });

    foo({ async fn baz(f: &u8) -> u8 { *f } baz });
}
