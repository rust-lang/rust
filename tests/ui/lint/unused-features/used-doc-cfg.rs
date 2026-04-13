//@ check-pass
//@ compile-flags: --check-cfg=cfg(feature,values("enabled_feature"))
#![crate_type = "lib"]
#![deny(unused_features)]
#![feature(doc_cfg)]

#[cfg(feature = "enabled_feature")]
pub fn foo() {}
