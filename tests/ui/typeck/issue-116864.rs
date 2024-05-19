//@ compile-flags: -Znext-solver
//@ check-pass
//@ edition: 2021

use std::future::Future;

trait Baz {
    type Param;
}

trait FnMutFut<P, R>: FnMut(P) -> Self::Future {
    type Future: Future<Output = R>;
}

impl<P, F, FUT, R> FnMutFut<P, R> for F
where
    F: FnMut(P) -> FUT,
    FUT: Future<Output = R>,
{
    type Future = FUT;
}

async fn foo<BAZ>(_: BAZ, mut cb: impl for<'any> FnMutFut<&'any BAZ::Param, ()>)
where
    BAZ: Baz<Param = i32>,
{
    cb(&1i32).await;
}

fn main() {
}
