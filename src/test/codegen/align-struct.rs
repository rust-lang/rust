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

#![feature(attr_literals)]
#![feature(repr_align)]

#[repr(align(64))]
pub struct Align64(i32);

pub struct Nested64 {
    a: Align64,
    b: i32,
    c: i32,
    d: i8,
}

pub enum Enum64 {
    A(Align64),
    B(i32),
}

// CHECK-LABEL: @align64
#[no_mangle]
pub fn align64(i : i32) -> Align64 {
// CHECK: %a64 = alloca %Align64, align 64
// CHECK: call void @llvm.memcpy.{{.*}}(i8* %{{.*}}, i8* %{{.*}}, i{{[0-9]+}} 64, i32 64, i1 false)
    let a64 = Align64(i);
    a64
}

// CHECK-LABEL: @nested64
#[no_mangle]
pub fn nested64(a: Align64, b: i32, c: i32, d: i8) -> Nested64 {
// CHECK: %n64 = alloca %Nested64, align 64
// CHECK: %a = alloca %Align64, align 64
    let n64 = Nested64 { a, b, c, d };
    n64
}

// CHECK-LABEL: @enum64
#[no_mangle]
pub fn enum64(a: Align64) -> Enum64 {
// CHECK: %e64 = alloca %Enum64, align 64
// CHECK: %a = alloca %Align64, align 64
    let e64 = Enum64::A(a);
    e64
}
