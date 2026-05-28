//@ run-pass
//@ compile-flags:-C target-feature=-crt-static -Z unstable-options
//@ ignore-musl - requires changing the linker which is hard

#[cfg(not(target_feature = "crt-static"))]
fn main() {}
