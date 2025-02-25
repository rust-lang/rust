//@ edition: 2021
//@ only-x86_64

use std::future::Future;
use std::pin::Pin;

#[target_feature(enable = "sse2")]
fn target_feature() -> Pin<Box<dyn Future<Output = ()> + 'static>> {
    todo!()
}

fn test(f: impl AsyncFn()) {}

fn main() {
    test(target_feature); //~ ERROR the trait bound
}
