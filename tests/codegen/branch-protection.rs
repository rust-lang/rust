// Test that the correct module flags are emitted with different branch protection flags.

//@ revisions: bti pacret leaf bkey none
//@ needs-llvm-components: aarch64
//@ [bti] compile-flags: -Z branch-protection=bti
//@ [pacret] compile-flags: -Z branch-protection=pac-ret
//@ [leaf] compile-flags: -Z branch-protection=pac-ret,leaf
//@ [bkey] compile-flags: -Z branch-protection=pac-ret,b-key
//@ compile-flags: --target aarch64-unknown-linux-gnu

#![crate_type = "lib"]
#![feature(no_core, lang_items)]
#![no_core]

#[lang = "sized"]
trait Sized {}

// A basic test function.
pub fn test() {}

// CHECK-BTI: !"branch-target-enforcement", i32 1
// CHECK-BTI: !"sign-return-address", i32 0
// CHECK-BTI: !"sign-return-address-all", i32 0
// CHECK-BTI: !"sign-return-address-with-bkey", i32 0

// CHECK-PACRET: !"branch-target-enforcement", i32 0
// CHECK-PACRET: !"sign-return-address", i32 1
// CHECK-PACRET: !"sign-return-address-all", i32 0
// CHECK-PACRET: !"sign-return-address-with-bkey", i32 0

// CHECK-LEAF: !"branch-target-enforcement", i32 0
// CHECK-LEAF: !"sign-return-address", i32 1
// CHECK-LEAF: !"sign-return-address-all", i32 1
// CHECK-LEAF: !"sign-return-address-with-bkey", i32 0

// CHECK-BKEY: !"branch-target-enforcement", i32 0
// CHECK-BKEY: !"sign-return-address", i32 1
// CHECK-BKEY: !"sign-return-address-all", i32 0
// CHECK-BKEY: !"sign-return-address-with-bkey", i32 1

// CHECK-NONE-NOT: branch-target-enforcement
// CHECK-NONE-NOT: sign-return-address
// CHECK-NONE-NOT: sign-return-address-all
// CHECK-NONE-NOT: sign-return-address-with-bkey
