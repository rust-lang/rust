// Regression test for https://github.com/rust-lang/rust/issues/100973

// @is "$.index[*][?(@.name=='m1' && @.kind == 'module')].inner.is_stripped" true
// @set m1 = "$.index[*][?(@.name=='m1')].id"
mod m1 {}

// @is "$.index[*][?(@.inner.name=='m1' && @.kind=='import')].inner.id" $m1
pub use m1::*;
