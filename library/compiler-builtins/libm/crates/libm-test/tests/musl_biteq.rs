//! compare

// PowerPC tests are failing on LLVM 13: https://github.com/rust-lang/rust/issues/88520
#[cfg(not(target_arch = "powerpc64"))]
#[cfg(all(test, feature = "musl-bitwise-tests"))]
include!(concat!(env!("OUT_DIR"), "/musl-tests.rs"));
