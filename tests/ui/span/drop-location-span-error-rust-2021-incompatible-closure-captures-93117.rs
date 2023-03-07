// compile-flags: -Wrust-2021-incompatible-closure-captures

pub struct A {}

impl A {
    async fn create(path: impl AsRef<std::path::Path>)  {
    ;
    crate(move || {} ).await
    }
}

trait C{async fn new(val: T) {}  //~ ERROR this file contains an unclosed delimiter
