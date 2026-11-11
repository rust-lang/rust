//@ check-pass
//@ compile-flags: --check-cfg=cfg(feature,values("enabled_feature"))
// Regression test for https://github.com/rust-lang/rust/issues/154487

#![crate_type = "lib"]
#![deny(unused_features)]
#![feature(doc_cfg)]

#[cfg(feature = "enabled_feature")]
pub fn foo() {}
