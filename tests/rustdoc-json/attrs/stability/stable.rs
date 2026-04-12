#![feature(staged_api)]

//@ is "$.index[?(@.name=='foo')].stability.level" '"stable"'
//@ is "$.index[?(@.name=='foo')].stability.feature" '"eeeee"'
//@ is "$.index[?(@.name=='foo')].attrs" []
#[stable(since = "2.71.8", feature = "eeeee")]
pub fn foo() {}
