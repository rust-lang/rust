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

#![feature(allocator)]

pub struct S {
  _field: [i64; 4],
}

pub struct UnsafeInner {
  _field: std::cell::UnsafeCell<i16>,
}

// CHECK: zeroext i1 @boolean(i1 zeroext)
#[no_mangle]
pub fn boolean(x: bool) -> bool {
  x
}

// CHECK: @readonly_borrow(i32* noalias readonly dereferenceable(4))
// FIXME #25759 This should also have `nocapture`
#[no_mangle]
pub fn readonly_borrow(_: &i32) {
}

// CHECK: @static_borrow(i32* noalias readonly dereferenceable(4))
// static borrow may be captured
#[no_mangle]
pub fn static_borrow(_: &'static i32) {
}

// CHECK: @named_borrow(i32* noalias readonly dereferenceable(4))
// borrow with named lifetime may be captured
#[no_mangle]
pub fn named_borrow<'r>(_: &'r i32) {
}

// CHECK: @unsafe_borrow(%UnsafeInner* dereferenceable(2))
// unsafe interior means this isn't actually readonly and there may be aliases ...
#[no_mangle]
pub fn unsafe_borrow(_: &UnsafeInner) {
}

// CHECK: @mutable_unsafe_borrow(%UnsafeInner* noalias dereferenceable(2))
// ... unless this is a mutable borrow, those never alias
#[no_mangle]
pub fn mutable_unsafe_borrow(_: &mut UnsafeInner) {
}

// CHECK: @mutable_borrow(i32* noalias dereferenceable(4))
// FIXME #25759 This should also have `nocapture`
#[no_mangle]
pub fn mutable_borrow(_: &mut i32) {
}

// CHECK: @indirect_struct(%S* noalias nocapture dereferenceable(32))
#[no_mangle]
pub fn indirect_struct(_: S) {
}

// CHECK: @borrowed_struct(%S* noalias readonly dereferenceable(32))
// FIXME #25759 This should also have `nocapture`
#[no_mangle]
pub fn borrowed_struct(_: &S) {
}

// CHECK: noalias dereferenceable(4) i32* @_box(i32* noalias dereferenceable(4))
#[no_mangle]
pub fn _box(x: Box<i32>) -> Box<i32> {
  x
}

// CHECK: @struct_return(%S* noalias nocapture sret dereferenceable(32))
#[no_mangle]
pub fn struct_return() -> S {
  S {
    _field: [0, 0, 0, 0]
  }
}

// CHECK: noalias i8* @allocator()
#[no_mangle]
#[allocator]
pub fn allocator() -> *const i8 {
  std::ptr::null()
}
