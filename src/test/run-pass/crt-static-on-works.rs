#![allow(stable_features)]
// compile-flags:-C target-feature=+crt-static -Z unstable-options

#![feature(cfg_target_feature)]

#[cfg(target_feature = "crt-static")]
fn main() {}
