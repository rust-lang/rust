// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(const_err)] // make sure we cannot allow away the errors tested here

#[repr(usize)]
#[derive(Copy, Clone)]
enum Enum {
    A = 0,
}
union TransmuteEnum {
    a: &'static u8,
    out: Enum,
}

// A pointer is guaranteed non-null
const BAD_ENUM: Enum = unsafe { TransmuteEnum { a: &1 }.out };
//~^ ERROR is undefined behavior

// (Potentially) invalid enum discriminant
#[repr(usize)]
#[derive(Copy, Clone)]
enum Enum2 {
    A = 2,
}
#[repr(transparent)]
#[derive(Copy, Clone)]
struct Wrap<T>(T);
union TransmuteEnum2 {
    in1: usize,
    in2: &'static u8,
    in3: (),
    out1: Enum2,
    out2: Wrap<Enum2>, // something wrapping the enum so that we test layout first, not enum
}
const BAD_ENUM2: Enum2 = unsafe { TransmuteEnum2 { in1: 0 }.out1 };
//~^ ERROR is undefined behavior
const BAD_ENUM3: Enum2 = unsafe { TransmuteEnum2 { in2: &0 }.out1 };
//~^ ERROR is undefined behavior
const BAD_ENUM4: Wrap<Enum2> = unsafe { TransmuteEnum2 { in2: &0 }.out2 };
//~^ ERROR is undefined behavior

// Undef enum discriminant. In an arry to avoid `Scalar` layout.
const BAD_ENUM_UNDEF: [Enum2; 2] = [unsafe { TransmuteEnum2 { in3: () }.out1 }; 2];
//~^ ERROR is undefined behavior

// Invalid enum field content (mostly to test printing of paths for enum tuple
// variants and tuples).
union TransmuteChar {
    a: u32,
    b: char,
}
// Need to create something which does not clash with enum layout optimizations.
const BAD_ENUM_CHAR: Option<(char, char)> = Some(('x', unsafe { TransmuteChar { a: !0 }.b }));
//~^ ERROR is undefined behavior

fn main() {
}
