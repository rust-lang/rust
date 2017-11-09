// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(arbitrary_self_types)]

use std::ptr;

trait Foo {
    fn foo(self: *const Self) -> &'static str;

    unsafe fn bar(self: *const Self) -> i64;
}

impl Foo for i32 {
    fn foo(self: *const Self) -> &'static str {
        "I'm an i32!"
    }

    unsafe fn bar(self: *const Self) -> i64 {
        *self as i64
    }
}

impl Foo for u32 {
    fn foo(self: *const Self) -> &'static str {
        "I'm a u32!"
    }

    unsafe fn bar(self: *const Self) -> i64 {
        *self as i64
    }
}

fn main() {
    let foo_i32 = ptr::null::<i32>() as *const Foo;
    let foo_u32 = ptr::null::<u32>() as *const Foo;

    assert_eq!("I'm an i32!", foo_i32.foo());
    assert_eq!("I'm a u32!", foo_u32.foo());

    let bar_i32 = 5i32;
    let bar_i32_thin = &bar_i32 as *const i32;
    assert_eq!(5, unsafe { bar_i32_thin.bar() });
    let bar_i32_fat = bar_i32_thin as *const Foo;
    assert_eq!(5, unsafe { bar_i32_fat.bar() });

    let bar_u32 = 18u32;
    let bar_u32_thin = &bar_u32 as *const u32;
    assert_eq!(18, unsafe { bar_u32_thin.bar() });
    let bar_u32_fat = bar_u32_thin as *const Foo;
    assert_eq!(18, unsafe { bar_u32_fat.bar() });
}
