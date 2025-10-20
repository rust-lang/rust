//@ compile-flags: -Copt-level=3 -Cno-prepopulate-passes

#![crate_type = "lib"]
#![allow(internal_features)]
#![feature(unsized_fn_params)]

use std::cell::Cell;
use std::hint;

// Check to make sure that we can deduce the `readonly` attribute from function bodies for
// parameters passed indirectly.

pub struct BigStruct {
    blah: [i32; 1024],
}

pub struct BigCellContainer {
    blah: [Cell<i32>; 1024],
}

// The by-value parameter for this big struct can be marked readonly.
//
// CHECK: @use_big_struct_immutably({{.*}} readonly {{.*}} %big_struct)
#[no_mangle]
pub fn use_big_struct_immutably(big_struct: BigStruct) {
    hint::black_box(&big_struct);
}

// The by-value parameter for this big struct can't be marked readonly, because we mutate it.
//
// CHECK: @use_big_struct_mutably(
// CHECK-NOT: readonly
// CHECK-SAME: %big_struct)
#[no_mangle]
pub fn use_big_struct_mutably(mut big_struct: BigStruct) {
    big_struct.blah[987] = 654;
    hint::black_box(&big_struct);
}

// The by-value parameter for this big struct can't be marked readonly, because it contains
// UnsafeCell.
//
// CHECK: @use_big_cell_container(
// CHECK-NOT: readonly
// CHECK-SAME: %big_cell_container)
#[no_mangle]
pub fn use_big_cell_container(big_cell_container: BigCellContainer) {
    hint::black_box(&big_cell_container);
}

// Make sure that we don't mistakenly mark a big struct as `readonly` when passed through a generic
// type parameter if it contains UnsafeCell.
//
// CHECK: @use_something(
// CHECK-NOT: readonly
// CHECK-SAME: %something)
#[no_mangle]
#[inline(never)]
pub fn use_something<T>(something: T) {
    hint::black_box(&something);
}

// Make sure that we still mark a big `Freeze` struct as `readonly` when passed through a generic
// type parameter.
//
// CHECK: @use_something_freeze(
// CHECK-SAME: readonly
// CHECK-SAME: %x)
#[no_mangle]
#[inline(never)]
pub fn use_something_freeze<T>(x: T) {}

#[no_mangle]
pub fn forward_big_cell_container(big_cell_container: BigCellContainer) {
    use_something(big_cell_container)
}

#[no_mangle]
pub fn forward_big_container(big_struct: BigStruct) {
    use_something_freeze(big_struct)
}
