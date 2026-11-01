//@ known-bug: #146210
//@ edition: 2024
//@ compile-flags: -Zvalidate-mir --crate-type lib
use core::pin::Pin;

fn bar<T>(non_send: T) -> Pin<Box<dyn Future<Output = ()> + Send>> {
    Box::pin(async {
        non_send;
    })
}
