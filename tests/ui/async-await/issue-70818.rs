//@ edition:2018

use std::future::Future;
fn foo<T: Send, U>(ty: T, ty1: U) -> impl Future<Output = (T, U)> + Send {
    //~^ ERROR future cannot be sent between threads safely
    async { (ty, ty1) }
}

fn main() {}
