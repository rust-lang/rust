// Regression test for <https://github.com/rust-lang/rust/issues/98658>

#![no_std]

//@ has "$.index[?(@.name=='Error')].inner.assoc_type"
//@ has "$.index[?(@.name=='Error')].inner.assoc_type.type" 10
//@ has "$.types[10].resolved_path"
//@ has "$.types[10].resolved_path.path" \"Infallible\"
pub struct ForBlanketTryFromImpl;
