// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-msvc

// compile-flags: -O -C no-prepopulate-passes

#![crate_type="lib"]

struct S;

impl Drop for S {
    fn drop(&mut self) {
    }
}

fn might_unwind() {
}

// CHECK-LABEL: @test
#[no_mangle]
pub fn test() {
    let _s = S;
    // Check that the personality slot alloca gets a lifetime start in each cleanup block, not just
    // in the first one.
    // CHECK-LABEL: cleanup:
    // CHECK: bitcast{{.*}}personalityslot
    // CHECK-NEXT: call void @llvm.lifetime.start
    // CHECK-LABEL: cleanup1:
    // CHECK: bitcast{{.*}}personalityslot
    // CHECK-NEXT: call void @llvm.lifetime.start
    might_unwind();
    let _t = S;
    might_unwind();
}
