// ignore-tidy-linelength
//@ revisions: ENABLE_SVE DISABLE_SVE DISABLE_NEON ENABLE_NEON
//@ compile-flags: --crate-type=rlib --target=aarch64-unknown-linux-gnu
//@ needs-llvm-components: aarch64

// The "+v8a" feature is matched as optional as it isn't added when we
// are targeting older LLVM versions. Once the min supported version
// is LLVM-14 we can remove the optional regex matching for this feature.

//@ [ENABLE_SVE] compile-flags: -C target-feature=+sve -Copt-level=0
// ENABLE_SVE: attributes #0 = { {{.*}} "target-features"="{{((\+outline-atomics,?)|(\+v8a,?)?|(\+sve,?)|(\+neon,?))*}}" }

//@ [DISABLE_SVE] compile-flags: -C target-feature=-sve -Copt-level=0
// DISABLE_SVE: attributes #0 = { {{.*}} "target-features"="{{((\+outline-atomics,?)|(\+v8a,?)?|(-sve,?)|(\+neon,?))*}}" }

//@ [DISABLE_NEON] compile-flags: -C target-feature=-neon -Copt-level=0
// DISABLE_NEON: attributes #0 = { {{.*}} "target-features"="{{((\+outline-atomics,?)|(\+v8a,?)?|(-fp-armv8,?)|(-neon,?))*}}" }

//@ [ENABLE_NEON] compile-flags: -C target-feature=+neon -Copt-level=0
// ENABLE_NEON: attributes #0 = { {{.*}} "target-features"="{{((\+outline-atomics,?)|(\+v8a,?)?|(\+fp-armv8,?)|(\+neon,?))*}}" }


#![feature(no_core, lang_items)]
#![no_core]

#[lang = "sized"]
trait Sized {}

pub fn test() {}
