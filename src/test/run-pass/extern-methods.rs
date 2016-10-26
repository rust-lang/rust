// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-arm
// ignore-aarch64

trait A {
    extern "fastcall" fn test1(i: i32);
    extern fn test2(i: i32);
}

struct S;
impl S {
    extern "stdcall" fn test3(i: i32) {
        assert_eq!(i, 3);
    }
}

impl A for S {
    extern "fastcall" fn test1(i: i32) {
        assert_eq!(i, 1);
    }
    extern fn test2(i: i32) {
        assert_eq!(i, 2);
    }
}

fn main() {
    <S as A>::test1(1);
    <S as A>::test2(2);
    S::test3(3);
}
