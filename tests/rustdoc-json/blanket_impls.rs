// Regression test for <https://github.com/rust-lang/rust/issues/98658>

#![no_std]

//@ jq_is '[.index[] | select(.name == "Error").inner | has("assoc_type")]' '[true, true]'
//@ jq_has '.index[] | select(.name == "Error").inner.assoc_type.type.resolved_path.path' '"Infallible"'
pub struct ForBlanketTryFromImpl;
