#![feature(staged_api)]

//@ is "$.index[?(@.name=='foo')].stability.level" '"unstable"'
//@ is "$.index[?(@.name=='foo')].stability.feature" '"delights"'
//@ is "$.index[?(@.name=='foo')].attrs" []
#[unstable(feature = "delights", issue = "26")]
pub fn foo() {}
