// Output = String caused an ICE whereas Output = &'static str compiled successfully.
// Broken MIR: coroutine contains type std::string::String in MIR,
// but typeck only knows about {<S as T>::Future, ()}
//@ check-pass
//@ edition:2018

use std::future::Future;

pub trait T {
    type Future: Future<Output = String>;
    fn bar() -> Self::Future;
}
pub async fn foo<S>() where S: T {
    S::bar().await;
    S::bar().await;
}
pub fn main() {}
