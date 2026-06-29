//@ min-llvm-version: 22
//@ compile-flags: -O -C no-prepopulate-passes

#![crate_type = "lib"]

use std::cell::UnsafeCell;

#[unsafe(no_mangle)]
pub fn store_ptr<'a>(x: *const i32, y: &mut *const i32) {
    // CHECK-LABEL: define {{.*}} @store_ptr
    // CHECK-NOT: !captures
    *y = x;
}

#[unsafe(no_mangle)]
pub fn store_ref_freeze<'a>(x: &'a i32, y: &mut &'a i32) {
    // CHECK-LABEL: define {{.*}} @store_ref_freeze
    // CHECK: store ptr %x, ptr %y, {{.*}}, !captures ![[CAPTURES:[0-9]+]]
    *y = x;
}

#[unsafe(no_mangle)]
pub fn store_ref_not_freeze<'a>(x: &'a UnsafeCell<i32>, y: &mut &'a UnsafeCell<i32>) {
    // CHECK-LABEL: define {{.*}} @store_ref_not_freeze
    // CHECK-NOT: !captures
    *y = x;
}

#[unsafe(no_mangle)]
pub fn store_mut_ref<'a>(x: &'a mut i32, y: &mut &'a mut i32) {
    // CHECK-LABEL: define {{.*}} @store_mut_ref
    // CHECK-NOT: !captures
    *y = x;
}

#[unsafe(no_mangle)]
pub fn store_mut_ref_as_shared_ref<'a>(x: &'a mut i32, y: &mut &'a i32) {
    // CHECK-LABEL: define {{.*}} @store_mut_ref_as_shared_ref
    // CHECK: store ptr %x, ptr %y, {{.*}}, !captures ![[CAPTURES:[0-9]+]]
    *y = x;
}

#[unsafe(no_mangle)]
pub fn store_mut_ref_as_ptr(x: &mut i32, y: &mut *const i32) {
    // CHECK-LABEL: define {{.*}} @store_mut_ref_as_ptr
    // CHECK-NOT: !captures
    *y = x;
}

#[unsafe(no_mangle)]
pub fn store_slice<'a>(x: &'a [i32], y: &mut &'a [i32]) {
    // CHECK-LABEL: define {{.*}} @store_slice
    // CHECK: store ptr %x.0, ptr %y, {{.*}}, !captures ![[CAPTURES:[0-9]+]]
    // Second store is slice size, can't have !captures
    // CHECK-NOT: !captures
    *y = x;
}

#[unsafe(no_mangle)]
pub fn store_dyn<'a>(x: &'a dyn Drop, y: &mut &'a dyn Drop) {
    // dyn trait is not known Freeze. The vtable could use !captures,
    // but that's probably not particularly useful.
    // CHECK-LABEL: define {{.*}} @store_dyn
    // CHECK-NOT: !captures
    *y = x;
}

// CHECK: ![[CAPTURES]] = !{!"address", !"read_provenance"}
