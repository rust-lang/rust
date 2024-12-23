//@ edition:2018
use std::future::Future;

async fn foo() {}

fn bar(f: impl Future<Output=()>) {}

fn main() {
    bar(foo); //~ERROR E0277
    let async_closure = async || ();
    bar(async_closure); //~ERROR E0277
}
