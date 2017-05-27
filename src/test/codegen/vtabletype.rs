// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This test depends on a patch that was committed to upstream LLVM
// after 5.0, then backported to the Rust LLVM fork.

// ignore-tidy-linelength
// ignore-windows
// ignore-macos
// min-system-llvm-version 5.1

// compile-flags: -g -C no-prepopulate-passes

// CHECK-LABEL: @main
// CHECK: {{.*}}DICompositeType{{.*}}name: "vtable",{{.*}}vtableHolder:{{.*}}

pub trait T {
}

impl T for f64 {
}

pub fn main() {
    let d = 23.0f64;
    let td = &d as &T;
}
