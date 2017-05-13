// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

const X: usize = 42 && 39; //~ ERROR E0080
                           //~| can't do this op on integrals
const ARR: [i32; X] = [99; 34]; //~ NOTE: for array length here

const X1: usize = 42 || 39; //~ ERROR E0080
                            //~| can't do this op on integrals
const ARR1: [i32; X1] = [99; 47]; //~ NOTE: for array length here

const X2: usize = -42 || -39; //~ ERROR E0080
                              //~| unary negation of unsigned integer
const ARR2: [i32; X2] = [99; 18446744073709551607]; //~ NOTE: for array length here

const X3: usize = -42 && -39; //~ ERROR E0080
                              //~| unary negation of unsigned integer
const ARR3: [i32; X3] = [99; 6]; //~ NOTE: for array length here

const Y: usize = 42.0 == 42.0;
const ARRR: [i32; Y] = [99; 1];
//~^ ERROR: expected `usize` for array length, found boolean [E0306]
//~| NOTE expected `usize`

const Y1: usize = 42.0 >= 42.0;
const ARRR1: [i32; Y] = [99; 1];
//~^ ERROR: expected `usize` for array length, found boolean [E0306]
//~| NOTE expected `usize`

const Y2: usize = 42.0 <= 42.0;
const ARRR2: [i32; Y] = [99; 1];
//~^ ERROR: expected `usize` for array length, found boolean [E0306]
//~| NOTE expected `usize`

const Y3: usize = 42.0 > 42.0;
const ARRR3: [i32; Y] = [99; 0];
//~^ ERROR: expected `usize` for array length, found boolean [E0306]
//~| NOTE expected `usize`

const Y4: usize = 42.0 < 42.0;
const ARRR4: [i32; Y] = [99; 0];
//~^ ERROR: expected `usize` for array length, found boolean [E0306]
//~| NOTE expected `usize`

const Y5: usize = 42.0 != 42.0;
const ARRR5: [i32; Y] = [99; 0];
//~^ ERROR: expected `usize` for array length, found boolean [E0306]
//~| NOTE expected `usize`


fn main() {
    let _ = ARR;
    let _ = ARRR;
    let _ = ARRR1;
    let _ = ARRR2;
    let _ = ARRR3;
    let _ = ARRR4;
    let _ = ARRR5;
}
