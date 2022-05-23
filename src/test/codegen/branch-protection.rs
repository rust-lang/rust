// Test that the correct module flags are emitted with different branch protection flags.

// revisions: BTI PACRET LEAF BKEY NONE
// min-llvm-version: 12.0.0
// needs-llvm-components: aarch64
// [BTI] compile-flags: -Z branch-protection=bti
// [PACRET] compile-flags: -Z branch-protection=pac-ret
// [LEAF] compile-flags: -Z branch-protection=pac-ret,leaf
// [BKEY] compile-flags: -Z branch-protection=pac-ret,b-key
// compile-flags: --target aarch64-unknown-linux-gnu

#![crate_type = "lib"]
#![feature(no_core, lang_items)]
#![no_core]

#[lang="sized"]
trait Sized { }

// A basic test function.
pub fn test() {
}

// BTI: !"branch-target-enforcement", i32 1
// BTI: !"sign-return-address", i32 0
// BTI: !"sign-return-address-all", i32 0
// BTI: !"sign-return-address-with-bkey", i32 0

// PACRET: !"branch-target-enforcement", i32 0
// PACRET: !"sign-return-address", i32 1
// PACRET: !"sign-return-address-all", i32 0
// PACRET: !"sign-return-address-with-bkey", i32 0

// LEAF: !"branch-target-enforcement", i32 0
// LEAF: !"sign-return-address", i32 1
// LEAF: !"sign-return-address-all", i32 1
// LEAF: !"sign-return-address-with-bkey", i32 0

// BKEY: !"branch-target-enforcement", i32 0
// BKEY: !"sign-return-address", i32 1
// BKEY: !"sign-return-address-all", i32 0
// BKEY: !"sign-return-address-with-bkey", i32 1

// NONE-NOT: branch-target-enforcement
// NONE-NOT: sign-return-address
// NONE-NOT: sign-return-address-all
// NONE-NOT: sign-return-address-with-bkey
