//@ edition: 2021

use std::pin::Pin;
use std::future::Future;

unsafe extern "Rust" {
    pub unsafe fn unsafety() -> Pin<Box<dyn Future<Output = ()> + 'static>>;
}

unsafe extern "C" {
    pub safe fn abi() -> Pin<Box<dyn Future<Output = ()> + 'static>>;
}

fn test(f: impl AsyncFn()) {}

fn main() {
    test(unsafety); //~ ERROR the trait bound
    test(abi); //~ ERROR the trait bound
}
