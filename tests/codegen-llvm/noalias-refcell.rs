//@ compile-flags: -Copt-level=3 -C no-prepopulate-passes -Z mutable-noalias=yes

#![crate_type = "lib"]

use std::cell::{Ref, RefCell, RefMut};

// Make sure that none of the arguments get a `noalias` attribute, because
// the `RefCell` might alias writes after either `Ref`/`RefMut` is dropped.

// CHECK-LABEL: @maybe_aliased(
// CHECK-NOT: noalias
// CHECK-SAME: %_refcell
#[no_mangle]
pub unsafe fn maybe_aliased(_: Ref<'_, i32>, _: RefMut<'_, i32>, _refcell: &RefCell<i32>) {}
