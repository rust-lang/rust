// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -C no-prepopulate-passes

#![crate_type = "lib"]

#[repr(packed)]
pub struct Packed {
    dealign: u8,
    data: u32
}

// CHECK-LABEL: @write_pkd
#[no_mangle]
pub fn write_pkd(pkd: &mut Packed) -> u32 {
// CHECK: %{{.*}} = load i32, i32* %{{.*}}, align 1
// CHECK: store i32 42, i32* %{{.*}}, align 1
    let result = pkd.data;
    pkd.data = 42;
    result
}
