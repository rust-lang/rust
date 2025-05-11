// Test that the correct module flags are emitted with different branch protection flags.

//@ add-core-stubs
//@ revisions: BTI PACRET LEAF BKEY PAUTHLR PAUTHLR_BKEY PAUTHLR_LEAF PAUTHLR_BTI NONE
//@ needs-llvm-components: aarch64
//@ [BTI] compile-flags: -Z branch-protection=bti
//@ [PACRET] compile-flags: -Z branch-protection=pac-ret
//@ [LEAF] compile-flags: -Z branch-protection=pac-ret,leaf
//@ [BKEY] compile-flags: -Z branch-protection=pac-ret,b-key
//@ [PAUTHLR] compile-flags: -Z branch-protection=pac-ret,pc
//@ [PAUTHLR_BKEY] compile-flags: -Z branch-protection=pac-ret,pc,b-key
//@ [PAUTHLR_LEAF] compile-flags: -Z branch-protection=pac-ret,pc,leaf
//@ [PAUTHLR_BTI] compile-flags: -Z branch-protection=bti,pac-ret,pc
//@ compile-flags: --target aarch64-unknown-linux-gnu

#![crate_type = "lib"]
#![feature(no_core, lang_items)]
#![no_core]

extern crate minicore;
use minicore::*;

// A basic test function.
// CHECK: @test(){{.*}} [[ATTR:#[0-9]+]] {
#[no_mangle]
pub fn test() {}

// BTI: attributes [[ATTR]] = {{.*}} "branch-target-enforcement"
// BTI: !"branch-target-enforcement", i32 1
// BTI: !"sign-return-address", i32 0
// BTI: !"branch-protection-pauth-lr", i32 0
// BTI: !"sign-return-address-all", i32 0
// BTI: !"sign-return-address-with-bkey", i32 0

// PACRET: attributes [[ATTR]] = {{.*}} "sign-return-address"="non-leaf"
// PACRET-SAME: "sign-return-address-key"="a_key"
// PACRET: !"branch-target-enforcement", i32 0
// PACRET: !"sign-return-address", i32 1
// PACRET: !"branch-protection-pauth-lr", i32 0
// PACRET: !"sign-return-address-all", i32 0
// PACRET: !"sign-return-address-with-bkey", i32 0

// LEAF: attributes [[ATTR]] = {{.*}} "sign-return-address"="all"
// LEAF-SAME: "sign-return-address-key"="a_key"
// LEAF: !"branch-target-enforcement", i32 0
// LEAF: !"sign-return-address", i32 1
// LEAF: !"branch-protection-pauth-lr", i32 0
// LEAF: !"sign-return-address-all", i32 1
// LEAF: !"sign-return-address-with-bkey", i32 0

// BKEY: attributes [[ATTR]] = {{.*}} "sign-return-address"="non-leaf"
// BKEY-SAME: "sign-return-address-key"="b_key"
// BKEY: !"branch-target-enforcement", i32 0
// BKEY: !"sign-return-address", i32 1
// BKEY: !"branch-protection-pauth-lr", i32 0
// BKEY: !"sign-return-address-all", i32 0
// BKEY: !"sign-return-address-with-bkey", i32 1

// PAUTHLR: attributes [[ATTR]] = {{.*}} "sign-return-address"="non-leaf"
// PAUTHLR-SAME: "sign-return-address-key"="a_key"
// PAUTHLR: !"branch-target-enforcement", i32 0
// PAUTHLR: !"sign-return-address", i32 1
// PAUTHLR: !"branch-protection-pauth-lr", i32 1
// PAUTHLR: !"sign-return-address-all", i32 0
// PAUTHLR: !"sign-return-address-with-bkey", i32 0

// PAUTHLR_BKEY: attributes [[ATTR]] = {{.*}} "sign-return-address"="non-leaf"
// PAUTHLR_BKEY-SAME: "sign-return-address-key"="b_key"
// PAUTHLR_BKEY: !"branch-target-enforcement", i32 0
// PAUTHLR_BKEY: !"sign-return-address", i32 1
// PAUTHLR_BKEY: !"branch-protection-pauth-lr", i32 1
// PAUTHLR_BKEY: !"sign-return-address-all", i32 0
// PAUTHLR_BKEY: !"sign-return-address-with-bkey", i32 1

// PAUTHLR_LEAF: attributes [[ATTR]] = {{.*}} "sign-return-address"="all"
// PAUTHLR_LEAF-SAME: "sign-return-address-key"="a_key"
// PAUTHLR_LEAF: !"branch-target-enforcement", i32 0
// PAUTHLR_LEAF: !"sign-return-address", i32 1
// PAUTHLR_LEAF: !"branch-protection-pauth-lr", i32 1
// PAUTHLR_LEAF: !"sign-return-address-all", i32 1
// PAUTHLR_LEAF: !"sign-return-address-with-bkey", i32 0

// PAUTHLR_BTI: attributes [[ATTR]] = {{.*}} "sign-return-address"="non-leaf"
// PAUTHLR_BTI-SAME: "sign-return-address-key"="a_key"
// PAUTHLR_BTI: !"branch-target-enforcement", i32 1
// PAUTHLR_BTI: !"sign-return-address", i32 1
// PAUTHLR_BTI: !"branch-protection-pauth-lr", i32 1
// PAUTHLR_BTI: !"sign-return-address-all", i32 0
// PAUTHLR_BTI: !"sign-return-address-with-bkey", i32 0

// NONE-NOT: branch-target-enforcement
// NONE-NOT: sign-return-address
// NONE-NOT: sign-return-address-all
// NONE-NOT: sign-return-address-with-bkey
