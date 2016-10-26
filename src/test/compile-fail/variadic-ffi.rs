// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern "cdecl" {
    fn printf(_: *const u8, ...); //~ ERROR: variadic function must have C calling convention
}

extern {
    fn foo(f: isize, x: u8, ...);
}

extern "C" fn bar(f: isize, x: u8) {}

fn main() {
    // errors below are no longer checked because error above aborts
    // compilation; see variadic-ffi-3.rs for corresponding test.
    unsafe {
        foo();
        foo(1);

        let x: unsafe extern "C" fn(f: isize, x: u8) = foo;
        let y: extern "C" fn(f: isize, x: u8, ...) = bar;

        foo(1, 2, 3f32);
        foo(1, 2, true);
        foo(1, 2, 1i8);
        foo(1, 2, 1u8);
        foo(1, 2, 1i16);
        foo(1, 2, 1u16);
    }
}
