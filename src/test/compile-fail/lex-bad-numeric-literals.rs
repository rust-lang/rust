// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn main() {
    0o1.0; //~ ERROR: octal float literal is not supported
    0o2f32; //~ ERROR: octal float literal is not supported
    0o3.0f32; //~ ERROR: octal float literal is not supported
    0o4e4; //~ ERROR: octal float literal is not supported
    0o5.0e5; //~ ERROR: octal float literal is not supported
    0o6e6f32; //~ ERROR: octal float literal is not supported
    0o7.0e7f64; //~ ERROR: octal float literal is not supported
    0x8.0e+9; //~ ERROR: hexadecimal float literal is not supported
    0x9.0e-9; //~ ERROR: hexadecimal float literal is not supported
}

static F: f32 =
    1e+ //~ ERROR: scan_exponent: bad fp literal
;


static F: f32 =
    0x539.0 //~ ERROR: hexadecimal float literal is not supported
;

static I: int =
    99999999999999999999999999999999 //~ ERROR: int literal is too large
;

static J: int =
    99999999999999999999999999999999u32 //~ ERROR: int literal is too large
;

static A: int =
    0x //~ ERROR: no valid digits
;
static B: int =
    0xu32 //~ ERROR: no valid digits
;
static C: int =
    0ou32 //~ ERROR: no valid digits
;
static D: int =
    0bu32 //~ ERROR: no valid digits
;
static E: int =
    0b //~ ERROR: no valid digits
;
static F: int =
    0o //~ ERROR: no valid digits
;
