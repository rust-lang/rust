// edition:2018
use std::future::Future;

async fn foo() {}

fn bar(f: impl Future<Output=()>) {}

fn main() {
    bar(foo); //~ERROR E0277
}
