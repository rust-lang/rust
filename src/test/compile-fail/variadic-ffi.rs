// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern "stdcall" {
    fn printf(_: *u8, ...); //~ ERROR: variadic function must have C calling convention
}

extern {
    fn foo(f: int, x: u8, ...);
}

extern "C" fn bar(f: int, x: u8) {}

fn main() {
    unsafe {
        foo(); //~ ERROR: this function takes at least 2 parameters but 0 parameters were supplied
        foo(1); //~ ERROR: this function takes at least 2 parameters but 1 parameter was supplied

        let x: extern "C" unsafe fn(f: int, x: u8) = foo;
        //~^ ERROR: mismatched types: expected `extern "C" unsafe fn(int, u8)` but found `extern "C" unsafe fn(int, u8, ...)` (expected non-variadic fn but found variadic function)

        let y: extern "C" unsafe fn(f: int, x: u8, ...) = bar;
        //~^ ERROR: mismatched types: expected `extern "C" unsafe fn(int, u8, ...)` but found `extern "C" extern fn(int, u8)` (expected variadic fn but found non-variadic function)

        foo(1, 2, 3f32); //~ ERROR: can't pass an f32 to variadic function, cast to c_double
        foo(1, 2, true); //~ ERROR: can't pass bool to variadic function, cast to c_int
        foo(1, 2, 1i8); //~ ERROR: can't pass i8 to variadic function, cast to c_int
        foo(1, 2, 1u8); //~ ERROR: can't pass u8 to variadic function, cast to c_uint
        foo(1, 2, 1i16); //~ ERROR: can't pass i16 to variadic function, cast to c_int
        foo(1, 2, 1u16); //~ ERROR: can't pass u16 to variadic function, cast to c_uint
    }
}
