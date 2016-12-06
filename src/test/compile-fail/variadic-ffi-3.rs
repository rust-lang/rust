// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern {
    fn foo(f: isize, x: u8, ...);
    //~^ defined here
    //~| defined here
}

extern "C" fn bar(f: isize, x: u8) {}

fn main() {
    unsafe {
        foo(); //~ ERROR: this function takes at least 2 parameters but 0 parameters were supplied
               //~| NOTE expected at least 2 parameters
        foo(1); //~ ERROR: this function takes at least 2 parameters but 1 parameter was supplied
                //~| NOTE expected at least 2 parameters

        let x: unsafe extern "C" fn(f: isize, x: u8) = foo;
        //~^ ERROR: mismatched types
        //~| expected type `unsafe extern "C" fn(isize, u8)`
        //~| found type `unsafe extern "C" fn(isize, u8, ...) {foo}`
        //~| NOTE: expected non-variadic fn, found variadic function

        let y: extern "C" fn(f: isize, x: u8, ...) = bar;
        //~^ ERROR: mismatched types
        //~| expected type `extern "C" fn(isize, u8, ...)`
        //~| found type `extern "C" fn(isize, u8) {bar}`
        //~| NOTE: expected variadic fn, found non-variadic function

        foo(1, 2, 3f32); //~ ERROR: can't pass an `f32` to variadic function, cast to `c_double`
        foo(1, 2, true); //~ ERROR: can't pass `bool` to variadic function, cast to `c_int`
        foo(1, 2, 1i8); //~ ERROR: can't pass `i8` to variadic function, cast to `c_int`
        foo(1, 2, 1u8); //~ ERROR: can't pass `u8` to variadic function, cast to `c_uint`
        foo(1, 2, 1i16); //~ ERROR: can't pass `i16` to variadic function, cast to `c_int`
        foo(1, 2, 1u16); //~ ERROR: can't pass `u16` to variadic function, cast to `c_uint`
    }
}
