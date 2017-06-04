// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-x86
// ignore-arm
// ignore-emscripten
// ignore 32-bit platforms (test output is different)

// Tests that `transmute` cannot be called on type parameters.

use std::mem::transmute;

unsafe fn f<T>(x: T) {
    let _: i32 = transmute(x);
//~^ ERROR differently sized types: T (size can vary) to i32
}

unsafe fn g<T>(x: (T, i32)) {
    let _: i32 = transmute(x);
//~^ ERROR differently sized types: (T, i32) (size can vary because of T) to i32
}

unsafe fn h<T>(x: [T; 10]) {
    let _: i32 = transmute(x);
//~^ ERROR differently sized types: [T; 10] (size can vary because of T) to i32
}

struct Bad<T> {
    f: T,
}

unsafe fn i<T>(x: Bad<T>) {
    let _: i32 = transmute(x);
//~^ ERROR differently sized types: Bad<T> (size can vary because of T) to i32
}

enum Worse<T> {
    A(T),
    B,
}

unsafe fn j<T>(x: Worse<T>) {
    let _: i32 = transmute(x);
//~^ ERROR differently sized types: Worse<T> (size can vary because of T) to i32
}

unsafe fn k<T>(x: Option<T>) {
    let _: i32 = transmute(x);
//~^ ERROR differently sized types: std::option::Option<T> (size can vary because of T) to i32
}

fn main() {}
