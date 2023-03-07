// edition: 2021

use std::future::Future;
fn foo() -> impl Future<Output=()> {
    async { }
}

fn main() {
    let fut = foo();
    fut.poll();
    //~^ ERROR no method named `poll` found for opaque type `impl Future<Output = ()>` in the current scope [E0599]
}
