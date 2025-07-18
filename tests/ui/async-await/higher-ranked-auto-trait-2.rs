// Repro for <https://github.com/rust-lang/rust/issues/111105#issue-1692860759>.
//@ edition: 2021
//@ revisions: assumptions no_assumptions
//@[assumptions] compile-flags: -Zhigher-ranked-assumptions
//@[assumptions] check-pass
//@[no_assumptions] known-bug: #110338

use std::future::Future;

pub trait Foo: Sync {
    fn run<'a, 'b: 'a, T: Sync>(&'a self, _: &'b T) -> impl Future<Output = ()> + 'a + Send;
}

pub trait FooExt: Foo {
    fn run_via<'a, 'b: 'a, T: Sync>(&'a self, t: &'b T) -> impl Future<Output = ()> + 'a + Send {
        async move {
            // asks for an unspecified lifetime to outlive itself? weird diagnostics
            self.run(t).await;
        }
    }
}

fn main() {}
