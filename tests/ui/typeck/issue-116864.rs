//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//@ edition: 2021

// This was fixed by lazy norm of param env with the next solver.
// But it regressed again as we switched back to be consistent with
// the old solver. See #158643.


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
//[next]~^ ERROR: expected an `FnOnce(&'any i32)` closure, found `impl for<'any> FnMutFut<&'any BAZ::Param, ()>`
//[current]~^^ ERROR: expected an `FnMut(&'any i32)` closure, found `impl for<'any> FnMutFut<&'any BAZ::Param, ()>`
//[current]~| ERROR: expected an `FnMut(&'any i32)` closure, found `impl for<'any> FnMutFut<&'any BAZ::Param, ()>`
//[current]~| ERROR: expected an `FnMut(&'any i32)` closure, found `impl for<'any> FnMutFut<&'any BAZ::Param, ()>`
where
    BAZ: Baz<Param = i32>,
{
    cb(&1i32).await;
    //[current]~^ ERROR: expected an `FnMut(&i32)` closure, found `impl for<'any> FnMutFut<&'any BAZ::Param, ()>`
    //[current]~| ERROR: mismatched types
}

fn main() {
}
