//@ compile-flags: -Znext-solver
//@ check-pass
//@ edition: 2024

// Ensure we don't end up in a query cycle due to trying to assemble an impl candidate
// for an RPITIT normalizes-to goal, even though that impl candidate would *necessarily*
// be made rigid by a where clause. This query cycle is thus avoidable by not assembling
// that impl candidate which we *know* we are going to throw away anyways.

use std::future::Future;

pub trait ReactiveFunction: Send {
    type Output;

    fn invoke(self) -> Self::Output;
}

trait AttributeValue {
    fn resolve(self) -> impl Future<Output = ()> + Send;
}

impl<F, V> AttributeValue for F
where
    F: ReactiveFunction<Output = V>,
    V: AttributeValue,
{
    async fn resolve(self) {
        self.invoke().resolve().await
    }
}

fn main() {}
