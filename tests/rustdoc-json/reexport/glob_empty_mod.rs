// Regression test for https://github.com/rust-lang/rust/issues/100973

// @is "$.index[*][?(@.name=='m1' && @.inner.module)].inner.module.is_stripped" true
// @set m1 = "$.index[*][?(@.name=='m1')].id"
mod m1 {}

// @is "$.index[*][?(@.inner.import)].inner.import.id" $m1
pub use m1::*;
