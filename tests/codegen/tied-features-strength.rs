// ignore-tidy-linelength
//@ add-core-stubs
//@ revisions: ENABLE_SVE DISABLE_SVE DISABLE_NEON ENABLE_NEON
//@ compile-flags: --crate-type=rlib --target=aarch64-unknown-linux-gnu
//@ needs-llvm-components: aarch64

// The "+fpmr" feature is matched as optional as it is only an explicit
// feature in LLVM 18. Once the min supported version is LLVM-19 the optional
// regex matching for this feature can be removed.

//@ [ENABLE_SVE] compile-flags: -C target-feature=+sve -Copt-level=0
// ENABLE_SVE: attributes #0 = { {{.*}} "target-features"="{{((\+outline-atomics,?)|(\+v8a,?)|(\+fpmr,?)?|(\+sve,?)|(\+neon,?)|(\+fp-armv8,?))*}}" }

//@ [DISABLE_SVE] compile-flags: -C target-feature=-sve -Copt-level=0
// DISABLE_SVE: attributes #0 = { {{.*}} "target-features"="{{((\+outline-atomics,?)|(\+v8a,?)|(\+fpmr,?)?|(-sve,?)|(\+neon,?))*}}" }

//@ [DISABLE_NEON] compile-flags: -C target-feature=-neon -Copt-level=0
// DISABLE_NEON: attributes #0 = { {{.*}} "target-features"="{{((\+outline-atomics,?)|(\+v8a,?)|(\+fpmr,?)?|(-fp-armv8,?)|(-neon,?))*}}" }

//@ [ENABLE_NEON] compile-flags: -C target-feature=+neon -Copt-level=0
// ENABLE_NEON: attributes #0 = { {{.*}} "target-features"="{{((\+outline-atomics,?)|(\+v8a,?)|(\+fpmr,?)?|(\+fp-armv8,?)|(\+neon,?))*}}" }

#![feature(no_core, lang_items)]
#![no_core]

extern crate minicore;
use minicore::*;

pub fn test() {}
