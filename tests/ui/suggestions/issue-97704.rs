//@ edition:2021

#![allow(unused)]

use std::future::Future;

async fn foo() -> Result<(), i32> {
    func(async { Ok::<_, i32>(()) })?;
    //~^ ERROR the trait bound `impl Future<Output = Result<(), i32>>: ops::try_trait_old::Try` is not satisfied [E0277]

    Ok(())
}

async fn func<T>(fut: impl Future<Output = T>) -> T {
    fut.await
}

fn main() {}
