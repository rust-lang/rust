//@ edition:2021
//@ run-rustfix

#![allow(unused)]

use std::future::Future;

async fn foo() -> Result<(), i32> {
    func(async { Ok::<_, i32>(()) })?;
    //~^ ERROR the `?` operator can only be applied to values that implement `Try`

    Ok(())
}

async fn func<T>(fut: impl Future<Output = T>) -> T {
    fut.await
}

fn main() {}
