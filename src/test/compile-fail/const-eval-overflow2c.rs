// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(unused_imports)]

// Note: the relevant lint pass here runs before some of the constant
// evaluation below (e.g. that performed by trans and llvm), so if you
// change this warn to a deny, then the compiler will exit before
// those errors are detected.

#![deny(const_err)]

use std::fmt;
use std::{i8, i16, i32, i64, isize};
use std::{u8, u16, u32, u64, usize};

const VALS_I8: (i8,) =
    (
     i8::MIN * 2,
     //~^ ERROR constant evaluation error
     //~| attempt to multiply with overflow
     );

const VALS_I16: (i16,) =
    (
     i16::MIN * 2,
     //~^ ERROR constant evaluation error
     //~| attempt to multiply with overflow
     );

const VALS_I32: (i32,) =
    (
     i32::MIN * 2,
     //~^ ERROR constant evaluation error
     //~| attempt to multiply with overflow
     );

const VALS_I64: (i64,) =
    (
     i64::MIN * 2,
     //~^ ERROR constant evaluation error
     //~| attempt to multiply with overflow
     );

const VALS_U8: (u8,) =
    (
     u8::MAX * 2,
     //~^ ERROR constant evaluation error
     //~| attempt to multiply with overflow
     );

const VALS_U16: (u16,) = (
     u16::MAX * 2,
     //~^ ERROR constant evaluation error
     //~| attempt to multiply with overflow
     );

const VALS_U32: (u32,) = (
     u32::MAX * 2,
     //~^ ERROR constant evaluation error
     //~| attempt to multiply with overflow
     );

const VALS_U64: (u64,) =
    (
     u64::MAX * 2,
     //~^ ERROR constant evaluation error
     //~| attempt to multiply with overflow
     );

fn main() {
    foo(VALS_I8);
    foo(VALS_I16);
    foo(VALS_I32);
    foo(VALS_I64);

    foo(VALS_U8);
    foo(VALS_U16);
    foo(VALS_U32);
    foo(VALS_U64);
}

fn foo<T>(_: T) {
}
