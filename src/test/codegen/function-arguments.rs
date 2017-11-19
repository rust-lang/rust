// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -C no-prepopulate-passes
// ignore-tidy-linelength

#![crate_type = "lib"]
#![feature(custom_attribute)]

pub struct S {
  _field: [i32; 8],
}

pub struct UnsafeInner {
  _field: std::cell::UnsafeCell<i16>,
}

// CHECK: zeroext i1 @boolean(i1 zeroext %x)
#[no_mangle]
pub fn boolean(x: bool) -> bool {
  x
}

// CHECK: @readonly_borrow(i32* noalias readonly dereferenceable(4) %arg0)
// FIXME #25759 This should also have `nocapture`
#[no_mangle]
pub fn readonly_borrow(_: &i32) {
}

// CHECK: @static_borrow(i32* noalias readonly dereferenceable(4) %arg0)
// static borrow may be captured
#[no_mangle]
pub fn static_borrow(_: &'static i32) {
}

// CHECK: @named_borrow(i32* noalias readonly dereferenceable(4) %arg0)
// borrow with named lifetime may be captured
#[no_mangle]
pub fn named_borrow<'r>(_: &'r i32) {
}

// CHECK: @unsafe_borrow(i16* dereferenceable(2) %arg0)
// unsafe interior means this isn't actually readonly and there may be aliases ...
#[no_mangle]
pub fn unsafe_borrow(_: &UnsafeInner) {
}

// CHECK: @mutable_unsafe_borrow(i16* dereferenceable(2) %arg0)
// ... unless this is a mutable borrow, those never alias
// ... except that there's this LLVM bug that forces us to not use noalias, see #29485
#[no_mangle]
pub fn mutable_unsafe_borrow(_: &mut UnsafeInner) {
}

// CHECK: @mutable_borrow(i32* dereferenceable(4) %arg0)
// FIXME #25759 This should also have `nocapture`
// ... there's this LLVM bug that forces us to not use noalias, see #29485
#[no_mangle]
pub fn mutable_borrow(_: &mut i32) {
}

// CHECK: @indirect_struct(%S* noalias nocapture dereferenceable(32) %arg0)
#[no_mangle]
pub fn indirect_struct(_: S) {
}

// CHECK: @borrowed_struct(%S* noalias readonly dereferenceable(32) %arg0)
// FIXME #25759 This should also have `nocapture`
#[no_mangle]
pub fn borrowed_struct(_: &S) {
}

// CHECK: noalias align 4 dereferenceable(4) i32* @_box(i32* noalias dereferenceable(4) %x)
#[no_mangle]
pub fn _box(x: Box<i32>) -> Box<i32> {
  x
}

// CHECK: @struct_return(%S* noalias nocapture sret dereferenceable(32))
#[no_mangle]
pub fn struct_return() -> S {
  S {
    _field: [0, 0, 0, 0, 0, 0, 0, 0]
  }
}

// Hack to get the correct size for the length part in slices
// CHECK: @helper([[USIZE:i[0-9]+]] %arg0)
#[no_mangle]
pub fn helper(_: usize) {
}

// CHECK: @slice([0 x i8]* noalias nonnull readonly %arg0.0, [[USIZE]] %arg0.1)
// FIXME #25759 This should also have `nocapture`
#[no_mangle]
pub fn slice(_: &[u8]) {
}

// CHECK: @mutable_slice([0 x i8]* nonnull %arg0.0, [[USIZE]] %arg0.1)
// FIXME #25759 This should also have `nocapture`
// ... there's this LLVM bug that forces us to not use noalias, see #29485
#[no_mangle]
pub fn mutable_slice(_: &mut [u8]) {
}

// CHECK: @unsafe_slice([0 x i16]* nonnull %arg0.0, [[USIZE]] %arg0.1)
// unsafe interior means this isn't actually readonly and there may be aliases ...
#[no_mangle]
pub fn unsafe_slice(_: &[UnsafeInner]) {
}

// CHECK: @str([0 x i8]* noalias nonnull readonly %arg0.0, [[USIZE]] %arg0.1)
// FIXME #25759 This should also have `nocapture`
#[no_mangle]
pub fn str(_: &[u8]) {
}

// CHECK: @trait_borrow(%"core::ops::drop::Drop"* nonnull %arg0.0, {}* noalias nonnull readonly %arg0.1)
// FIXME #25759 This should also have `nocapture`
#[no_mangle]
pub fn trait_borrow(_: &Drop) {
}

// CHECK: @trait_box(%"core::ops::drop::Drop"* noalias nonnull, {}* noalias nonnull readonly)
#[no_mangle]
pub fn trait_box(_: Box<Drop>) {
}

// CHECK: { [0 x i16]*, [[USIZE]] } @return_slice([0 x i16]* noalias nonnull readonly %x.0, [[USIZE]] %x.1)
#[no_mangle]
pub fn return_slice(x: &[u16]) -> &[u16] {
  x
}

// CHECK: noalias i8* @allocator()
#[no_mangle]
#[allocator]
pub fn allocator() -> *const i8 {
  std::ptr::null()
}
