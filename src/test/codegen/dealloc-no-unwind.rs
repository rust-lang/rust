// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
//
// no-system-llvm
// compile-flags: -O

#![crate_type="lib"]

struct A;

impl Drop for A {
    fn drop(&mut self) {
        extern { fn foo(); }
        unsafe { foo(); }
    }
}

#[no_mangle]
pub fn a(a: Box<i32>) {
    // CHECK-LABEL: define void @a
    // CHECK: call void @__rust_dealloc
    // CHECK-NEXT: call void @foo
    let _a = A;
    drop(a);
}
