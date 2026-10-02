//@ run-pass
//@ edition: 2021

use std::future::Future;
use std::pin::Pin;

#[expect(improper_ctypes_definitions)]
pub extern "C" fn abi() -> Pin<Box<dyn Future<Output = ()> + 'static>> {
    Box::pin(async {})
}

fn test(f: impl AsyncFn()) {
    let _ = async {
        f().await;
        f().await;
    };
}

fn main() {
    test(abi);
}
