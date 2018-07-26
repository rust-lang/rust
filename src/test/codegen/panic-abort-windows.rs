// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-tidy-linelength

// This test is for *-windows-msvc only.
// ignore-android
// ignore-bitrig
// ignore-cloudabi
// ignore-dragonfly
// ignore-emscripten
// ignore-freebsd
// ignore-haiku
// ignore-ios
// ignore-linux
// ignore-macos
// ignore-netbsd
// ignore-openbsd
// ignore-solaris

// compile-flags: -C no-prepopulate-passes -C panic=abort -O

#![crate_type = "lib"]

// CHECK: Function Attrs: nounwind uwtable
// CHECK-NEXT: define void @normal_uwtable()
#[no_mangle]
pub fn normal_uwtable() {
}

// CHECK: Function Attrs: nounwind uwtable
// CHECK-NEXT: define void @extern_uwtable()
#[no_mangle]
pub extern fn extern_uwtable() {
}
