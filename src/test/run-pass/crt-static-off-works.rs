#![allow(stable_features)]
// compile-flags:-C target-feature=-crt-static -Z unstable-options
// ignore-musl - requires changing the linker which is hard

#![feature(cfg_target_feature)]

#[cfg(not(target_feature = "crt-static"))]
fn main() {}
