// edition:2021
// revisions: current next
//[next] compile-flags: -Ztrait-solver=next

//[next]~^^^^ ERROR overflow evaluating the requirement

use std::future::Future;

pub trait Baz {
    type Param;
}

pub trait FnMutFut<P, R>: FnMut(P) -> Self::Future {
    type Future: Future<Output = R>;
}

impl<P, F, FUT, R> FnMutFut<P, R> for F
where
    F: FnMut(P) -> FUT,
    FUT: Future<Output = R>,
{
    type Future = FUT;
}

pub async fn does_not_work<BAZ>(_: BAZ, mut cb: impl for<'any> FnMutFut<&'any BAZ::Param, ()>)
//[current]~^ ERROR expected a `FnMut(&'any i32)` closure, found `imp
where
    BAZ: Baz<Param = i32>,
{
//[current]~^ ERROR expected a `FnMut(&'any i32)` closure, found
    cb(&1i32).await;
    //[current]~^ ERROR expected a `FnMut(&i32)` closure, found `impl for<'any> FnMutFut<&'
    //[current]~| ERROR mismatched types
    //[next]~^^^ ERROR type annotations needed
}

fn main() {
}
