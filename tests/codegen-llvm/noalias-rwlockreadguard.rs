//@ compile-flags: -Copt-level=3 -C no-prepopulate-passes -Z mutable-noalias=yes

#![crate_type = "lib"]

use std::sync::{RwLock, RwLockReadGuard};

// Make sure that `RwLockReadGuard` does not get a `noalias` attribute, because
// the `RwLock` might alias writes after it is dropped.

// CHECK-LABEL: @maybe_aliased(
// CHECK-NOT: noalias
// CHECK-SAME: %_data
#[no_mangle]
pub unsafe fn maybe_aliased(_: RwLockReadGuard<'_, i32>, _data: &RwLock<i32>) {}
