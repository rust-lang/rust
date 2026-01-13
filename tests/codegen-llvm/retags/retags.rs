// Verifies that retag intrinsics show up as expected with `-Zcodegen-emit-retag`.
//@ compile-flags: -Zcodegen-emit-retag -Copt-level=0

#![crate_type = "lib"]
#![feature(rustc_attrs)]
#![feature(allocator_api)]

use std::marker::PhantomPinned;
pub struct NotUnpin {
    _field: i32,
    _marker: PhantomPinned,
}

pub struct UnsafeInner {
    _field: std::cell::UnsafeCell<i16>,
}

// CHECK-LABEL: @readonly_borrow(ptr align {{.*}} %0)
#[no_mangle]
pub fn readonly_borrow(_: &i32) {
    // CHECK:       start:
    // CHECK-NEXT:  call ptr @__rust_retag_reg(ptr %0
}

// CHECK-LABEL: @mutable_borrow(ptr align {{.*}} %0)
#[no_mangle]
pub fn mutable_borrow(_: &mut i32) {
    // CHECK:       start:
    // CHECK-NEXT:  call ptr @__rust_retag_reg(ptr %0
}

// CHECK-LABEL: @option_borrow(ptr align {{.*}} %0)
#[no_mangle]
pub fn option_borrow(_x: Option<&i32>) {
    // CHECK: start:
    // CHECK: switch i64 %{{.+}}, label %[[V_T:.+]] [
    // CHECK-NEXT: i64 1, label %[[V:.+]]
    // CHECK-NEXT: ]
    // CHECK: [[V_T]]:
    // CHECK: phi ptr [ %0, %start ], [ %[[R:.+]], %[[V]] ]
    // CHECK: [[V]]:
    // CHECK-NEXT: %[[R]] = call ptr @__rust_retag_reg(ptr %0
    // CHECK: br label %[[V_T]]
}

// Retagging is a no-op for all `!Unpin`.
// CHECK-LABEL: @readonly_notunpin_borrow(ptr align {{.*}} %0
#[no_mangle]
pub fn readonly_notunpin_borrow(_: &NotUnpin) {
    // CHECK:       start:
    // CHECK-NEXT:  call ptr @__rust_retag_reg(ptr %0
}

// CHECK-LABEL: @mutable_notunpin_borrow
#[no_mangle]
pub fn mutable_notunpin_borrow(_: &mut NotUnpin) {
    // CHECK-NOT: call {{ptr|void}} @__rust_retag
}

enum E {
    A(&'static i8),
    B(&'static i32),
    C(&'static i64),
}

// CHECK-LABEL: @multiple_variants(i64 %_x.0, ptr %0
#[no_mangle]
pub fn multiple_variants(_x: E) {
    // CHECK: start:
    // CHECK-NEXT: switch i64 %_x.0, label %[[V_T:.+]] [
    // CHECK-NEXT: i64 0, label %[[V0:.+]]
    // CHECK-NEXT: i64 1, label %[[V1:.+]]
    // CHECK-NEXT: i64 2, label %[[V2:.+]]
    // CHECK-NEXT: ]
    // CHECK: [[V_T]]:
    // CHECK-NEXT: phi ptr [ %0, %start ], [ %[[R0:.+]], %[[V0]] ], [ %[[R1:.+]], %[[V1]] ], [ %[[R2:.+]], %[[V2]] ]
    // CHECK: [[V0]]:
    // CHECK-NEXT: %[[R0]] = call ptr @__rust_retag_reg(ptr %0, i64 1
    // CHECK: [[V1]]:
    // CHECK-NEXT: %[[R1]] = call ptr @__rust_retag_reg(ptr %0, i64 4
    // CHECK: [[V2]]:
    // CHECK-NEXT: %[[R2]] = call ptr @__rust_retag_reg(ptr %0, i64 8
}

// CHECK-LABEL: @_box(ptr align {{.*}} %0
#[no_mangle]
pub fn _box(x: Box<i32>) -> Box<i32> {
    // CHECK:       start:
    // CHECK-NEXT:  %[[R1:.+]] = call ptr @__rust_retag_reg(ptr %0
    // CHECK-NEXT:  %[[R2:.+]] = call ptr @__rust_retag_reg(ptr %[[R1]]
    // CHECK-NEXT:  ret ptr %[[R2]]
    x
}
// If a `Box` comes from the global allocator, then its innermost pointer
// should not be retagged, but we still want to retag the allocator.
// CHECK-LABEL: @_box_custom(ptr align {{.*}} %x.0, ptr %0)
#[no_mangle]
pub fn _box_custom(x: Box<i32, &std::alloc::Global>) {
    // CHECK: start:
    // CHECK-NEXT: call ptr @__rust_retag_reg(ptr %0
    drop(x)
}

// CHECK-LABEL: @slice(ptr %0
#[no_mangle]
pub fn slice(_: &[u8]) {
    // CHECK: start:
    // CHECK-NEXT: call ptr @__rust_retag_reg(ptr %0
}

// CHECK-LABEL: @mutable_slice(ptr %0
#[no_mangle]
pub fn mutable_slice(_: &mut [u8]) {
    // CHECK:       start:
    // CHECK-NEXT:  call ptr @__rust_retag_reg(ptr %0
}

// CHECK-LABEL: @unsafe_slice(ptr align {{.*}} %0
#[no_mangle]
pub fn unsafe_slice(_: &[UnsafeInner]) {
    // CHECK: start:
    // CHECK-NEXT: call ptr @__rust_retag_reg(ptr %0
}

// CHECK-LABEL: @str(ptr %0, i64 %_1.1)
#[no_mangle]
pub fn str(_: &[u8]) {
    // CHECK: start:
    // CHECK-NEXT: call ptr @__rust_retag_reg(ptr %0
}

// CHECK-LABEL: @return_slice(ptr align {{.*}} %0, i64 %x.1)
#[no_mangle]
pub fn return_slice(x: &[u16]) -> &[u16] {
    // CHECK: start:
    // CHECK-NEXT: %[[R1:.+]] = call ptr @__rust_retag_reg(ptr %0
    // CHECK-NEXT: call ptr @__rust_retag_reg(ptr %[[R1]]
    x
}

// CHECK-LABEL: @trait_borrow(ptr %0, ptr align {{.+}} %_1.1)
#[no_mangle]
pub fn trait_borrow(_: &dyn Drop) {
    // CHECK:       start:
    // CHECK-NEXT:  call ptr @__rust_retag_reg(ptr %0
}

// CHECK-LABEL: @trait_mutable_borrow
#[no_mangle]
pub fn trait_mutable_borrow(_: &mut dyn Drop) {
    // CHECK-NOT: call {{ptr|void}} @__rust_retag
}

// CHECK-LABEL: @option_trait_borrow(ptr %0, ptr %x.1)
#[no_mangle]
pub fn option_trait_borrow(x: Option<&dyn Drop>) {
    // CHECK: start:
    // CHECK: switch i64 %{{.+}}, label %v_t [
    // CHECK-NEXT: i64 1, label %v
    // CHECK-NEXT: ]
    // CHECK: v_t:
    // CHECK-NEXT: phi ptr [ %0, %start ], [ %[[R:.+]], %v ]
    // CHECK: v:
    // CHECK-NEXT: %[[R]] = call ptr @__rust_retag_reg(ptr %0
    // CHECK: br label %v_t
}

//CHECK-LABEL: @retag_mixed
#[no_mangle]
fn retag_mixed() {
    // CHECK: %{{.+}} = call ptr @__rust_retag_reg(ptr %{{.+}}, i64 4
    // CHECK: call void @__rust_retag_mem(ptr %target_alias, i64 4
    // CHECK-NEXT: %{{.+}} = call ptr @__rust_retag_reg(ptr %target_alias, i64 8
    let target = &mut 42;
    let mut target_alias = &42;
    retarget(&mut target_alias);

    #[no_mangle]
    fn retarget(_: &mut &u32) {}
}

//CHECK-LABEL: @option_trait_borrow_mut
#[no_mangle]
pub fn option_trait_borrow_mut(_: Option<&mut dyn Drop>) {
    // CHECK-NOT: call {{ptr|void}} @__rust_retag
}

//CHECK-LABEL: @trait_box
#[no_mangle]
pub fn trait_box(_: Box<dyn Drop + Unpin>) {
    // CHECK-NOT: call {{ptr|void}} @__rust_retag
}

//CHECK-LABEL: @trait_mutref
#[no_mangle]
pub fn trait_mutref(_: &mut (dyn Drop + Unpin)) {
    // CHECK-NOT: call {{ptr|void}} @__rust_retag
}

//CHECK-LABEL: @trait_option
#[no_mangle]
pub fn trait_option(x: Option<Box<dyn Drop + Unpin>>) -> Option<Box<dyn Drop + Unpin>> {
    // CHECK-NOT: call {{ptr|void}} @__rust_retag
    x
}
