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
// ignore-windows
// ignore-macos

// compile-flags: -g -C no-prepopulate-passes

#![feature(start)]

// CHECK-LABEL: @main
// CHECK: load volatile i8, i8* getelementptr inbounds ([[B:\[[0-9]* x i8\]]], [[B]]* @__rustc_debug_gdb_scripts_section__, i32 0, i32 0), align 1

#[start]
fn start(_: isize, _: *const *const u8) -> isize {
    return 0;
}
