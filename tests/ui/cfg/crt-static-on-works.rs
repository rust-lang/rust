//@run
//@compile-flags:-C target-feature=+crt-static
//@only-target-msvc

#[cfg(target_feature = "crt-static")]
fn main() {}
