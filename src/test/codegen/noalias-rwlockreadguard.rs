// compile-flags: -O -C no-prepopulate-passes -Z mutable-noalias=yes

#![crate_type = "lib"]

use std::cell::UnsafeCell;
use std::sync::RwLockReadGuard;

// Make sure that `RwLockReadGuard` does not get a `noalias` attribute, because
// the `UnsafeCell` might alias writes after it is dropped.

// CHECK-LABEL: @maybe_aliased(
// CHECK-NOT: noalias
#[no_mangle]
pub unsafe fn maybe_aliased(_: RwLockReadGuard<'_, i32>, _data: &UnsafeCell<i32>) {}
