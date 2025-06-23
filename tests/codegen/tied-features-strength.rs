// ignore-tidy-linelength
//@ add-core-stubs
//@ revisions: ENABLE_SVE DISABLE_SVE DISABLE_NEON ENABLE_NEON
//@ compile-flags: --crate-type=rlib --target=aarch64-unknown-linux-gnu
//@ needs-llvm-components: aarch64

// Rust made SVE require neon.
//@ [ENABLE_SVE] compile-flags: -C target-feature=+sve -Copt-level=0
// ENABLE_SVE: attributes #0
// ENABLE_SVE-SAME: +neon
// ENABLE_SVE-SAME: +sve

// However, disabling SVE does not disable neon.
//@ [DISABLE_SVE] compile-flags: -C target-feature=-sve -Copt-level=0
// DISABLE_SVE: attributes #0
// DISABLE_SVE-NOT: -neon
// DISABLE_SVE-SAME: -sve

// OTOH, neon fn `fp-armv8` are fully tied; toggling neon must toggle `fp-armv8` the same way.
//@ [DISABLE_NEON] compile-flags: -C target-feature=-neon -Copt-level=0
// DISABLE_NEON: attributes #0
// DISABLE_NEON-SAME: -neon
// DISABLE_NEON-SAME: -fp-armv8

//@ [ENABLE_NEON] compile-flags: -C target-feature=+neon -Copt-level=0
// ENABLE_NEON: attributes #0 = { {{.*}} "target-features"="{{((\+outline-atomics,?)|(\+v8a,?)|(\+fp-armv8,?)|(\+neon,?))*}}" }

#![feature(no_core, lang_items)]
#![no_core]

extern crate minicore;
use minicore::*;

pub fn test() {}
