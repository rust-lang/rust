//@ run-pass
//@ compile-flags:-C target-feature=+crt-static
//@ only-msvc

#[cfg(target_feature = "crt-static")]
fn main() {}
