// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:nounwind.rs
// compile-flags: -C no-prepopulate-passes -C panic=abort -C metadata=a
// ignore-windows

#![crate_type = "lib"]

extern crate nounwind;

#[no_mangle]
pub fn foo() {
    nounwind::bar();
// CHECK: @foo() unnamed_addr #0
// CHECK: @bar() unnamed_addr #0
// CHECK: attributes #0 = { {{.*}}nounwind{{.*}} }
}

