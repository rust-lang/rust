//@ run-rustfix
//@ edition:2021
use std::future::Future;
use std::pin::Pin;
pub struct S;
pub fn foo() {
    let _ = Box::pin(async move {
        if true {
            Ok(S) //~ ERROR mismatched types
        }
        Err(())
    });
}
pub fn bar() -> Pin<Box<dyn Future<Output = Result<S, ()>> + 'static>> {
    Box::pin(async move {
        if true {
            Ok(S) //~ ERROR mismatched types
        }
        Err(())
    })
}
fn main() {}
