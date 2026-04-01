//@ check-pass
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@ edition: 2024

// Regression test for <https://github.com/rust-lang/trait-system-refactor-initiative/issues/185>.
// Avoid unnecessarily computing the RPITIT type of the first impl when checking the WF of the
// second impl, since the first impl relies on the hidden type of the second impl.

use std::future::Future;

trait Handler {}

struct W<T>(T);

trait SendTarget {
    fn call(self) -> impl Future<Output = ()> + Send;
}

impl<T> SendTarget for W<T>
where
    T: Handler + Send,
{
    async fn call(self) {
        todo!()
    }
}

impl<T> SendTarget for T
where
    T: Handler + Send,
{
    async fn call(self) {
        W(self).call().await
    }
}

fn main() {}
