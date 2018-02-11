// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![deny(improper_ctypes)]
#![allow(dead_code)]

enum Z { }
enum U { A }
enum B { C, D }
enum T { E, F, G }

#[repr(C)]
enum ReprC { A, B, C }

#[repr(u8)]
enum U8 { A, B, C }

#[repr(isize)]
enum Isize { A, B, C }

extern {
   fn zf(x: Z);
   fn uf(x: U); //~ ERROR found enum without foreign-function-safe
   fn bf(x: B); //~ ERROR found enum without foreign-function-safe
   fn tf(x: T); //~ ERROR found enum without foreign-function-safe
   fn reprc(x: ReprC);
   fn u8(x: U8);
   fn isize(x: Isize);
}

pub fn main() { }
