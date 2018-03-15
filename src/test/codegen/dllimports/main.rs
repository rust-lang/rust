// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This test is for *-windows-msvc only.
// ignore-gnu
// ignore-android
// ignore-bitrig
// ignore-macos
// ignore-dragonfly
// ignore-freebsd
// ignore-haiku
// ignore-ios
// ignore-linux
// ignore-netbsd
// ignore-openbsd
// ignore-solaris
// ignore-emscripten

// aux-build:dummy.rs
// aux-build:wrapper.rs

extern crate wrapper;

// Check that external symbols coming from foreign dylibs are adorned with 'dllimport',
// whereas symbols coming from foreign staticlibs are not. (RFC-1717)

// CHECK: @dylib_global1 = external dllimport local_unnamed_addr global i32
// CHECK: @dylib_global2 = external dllimport local_unnamed_addr global i32
// CHECK: @static_global1 = external local_unnamed_addr global i32
// CHECK: @static_global2 = external local_unnamed_addr global i32

// CHECK: declare dllimport i32 @dylib_func1(i32)
// CHECK: declare dllimport i32 @dylib_func2(i32)
// CHECK: declare i32 @static_func1(i32)
// CHECK: declare i32 @static_func2(i32)

#[link(name = "dummy", kind="dylib")]
extern "C" {
    pub fn dylib_func1(x: i32) -> i32;
    pub static dylib_global1: i32;
}

#[link(name = "dummy", kind="static")]
extern "C" {
    pub fn static_func1(x: i32) -> i32;
    pub static static_global1: i32;
}

fn main() {
    unsafe {
        dylib_func1(dylib_global1);
        wrapper::dylib_func2(wrapper::dylib_global2);

        static_func1(static_global1);
        wrapper::static_func2(wrapper::static_global2);
    }
}
