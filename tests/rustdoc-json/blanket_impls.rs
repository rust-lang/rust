// Regression test for <https://github.com/rust-lang/rust/issues/98658>

#![no_std]

// @has "$.index[*][?(@.name=='Error')].inner.assoc_type"
// @has "$.index[*][?(@.name=='Error')].inner.assoc_type.default.resolved_path"
// @has "$.index[*][?(@.name=='Error')].inner.assoc_type.default.resolved_path.name" \"Infallible\"
pub struct ForBlanketTryFromImpl;
