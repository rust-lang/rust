fn main() {}

// This forces explicit emission of a +neon target feature on targets
// where it is implied by the target-cpu, like aarch64-apple-darwin.
// This is a regression test for #153397.
#[cfg(target_feature = "neon")]
#[target_feature(enable = "neon")]
#[unsafe(no_mangle)]
pub fn with_neon() {}
