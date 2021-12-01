// Test that the correct module flags are emitted with different branch protection flags.

// revisions: bti pac-ret leaf b-key
// min-llvm-version: 12.0.0
// needs-llvm-components: aarch64
// [bti] compile-flags: -Z branch-protection=bti
// [pac-ret] compile-flags: -Z branch-protection=pac-ret
// [leaf] compile-flags: -Z branch-protection=pac-ret,leaf
// [b-key] compile-flags: -Z branch-protection=pac-ret,b-key
// compile-flags: --target aarch64-unknown-linux-gnu

#![crate_type = "lib"]
#![feature(no_core, lang_items)]
#![no_core]

#[lang="sized"]
trait Sized { }

// A basic test function.
pub fn test() {
}

// bti: !"branch-target-enforcement", i32 1
// bti: !"sign-return-address", i32 0
// bti: !"sign-return-address-all", i32 0
// bti: !"sign-return-address-with-bkey", i32 0

// pac-ret: !"branch-target-enforcement", i32 0
// pac-ret: !"sign-return-address", i32 1
// pac-ret: !"sign-return-address-all", i32 0
// pac-ret: !"sign-return-address-with-bkey", i32 0

// leaf: !"branch-target-enforcement", i32 0
// leaf: !"sign-return-address", i32 1
// leaf: !"sign-return-address-all", i32 1
// leaf: !"sign-return-address-with-bkey", i32 0

// b-key: !"branch-target-enforcement", i32 0
// b-key: !"sign-return-address", i32 1
// b-key: !"sign-return-address-all", i32 0
// b-key: !"sign-return-address-with-bkey", i32 1
