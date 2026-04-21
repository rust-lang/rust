//@ compile-flags: --crate-type=lib --target x86_64-unknown-linux-gnu -O -Zinline-mir=no -C no-prepopulate-passes
//@ needs-llvm-components: x86
//@ only-x86_64
//@ ignore-backends: gcc

#![feature(core_intrinsics, target_feature_inline_always)]
#![allow(incomplete_features)]

use std::intrinsics::{AtomicOrdering, atomic_load};

#[inline(always)]
#[target_feature(enable = "cmpxchg16b")]
#[unsafe(no_mangle)]
pub fn load(x: *const u128) -> u128 {
    unsafe { atomic_load::<u128, { AtomicOrdering::Relaxed }>(x) }
}

#[unsafe(no_mangle)]
// CHECK-LABEL: define{{.*}} @load_core(
pub fn load_core(x: *const u128) -> u128 {
    // `cmpxchg16b` is not enabled on the caller, so the ineligible
    // `#[inline(always)]` callee must be marked `noinline` at the callsite.
    //
    // CHECK: %_0 = {{(tail )?}}call{{.*}} @load(ptr{{.*}} %x) [[CALL_ATTRS:#[0-9]+]]
    unsafe { load(x) }
}

// CHECK: attributes [[CALL_ATTRS]] = { {{.*}}noinline{{.*}} }
