// Regression test for <https://github.com/rust-lang/rust/issues/98658>

#![no_std]

//@ has "$.index[?(@.name=='Error')].inner.assoc_type"
//@ has "$.index[?(@.name=='Error')].inner.assoc_type.type.resolved_path"
//@ has "$.index[?(@.name=='Error')].inner.assoc_type.type.resolved_path.path" \"Infallible\"
pub struct ForBlanketTryFromImpl;
