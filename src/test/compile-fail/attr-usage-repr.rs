// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(dead_code)]
#![feature(repr_simd)]

#[repr(C)] //~ ERROR: attribute should be applied to struct, enum or union
fn f() {}

#[repr(C)]
struct SExtern(f64, f64);

#[repr(packed)]
struct SPacked(f64, f64);

#[repr(simd)]
struct SSimd(f64, f64);

#[repr(i8)] //~ ERROR: attribute should be applied to enum
struct SInt(f64, f64);

#[repr(C)]
enum EExtern { A, B }

#[repr(packed)] //~ ERROR: attribute should be applied to struct
enum EPacked { A, B }

#[repr(simd)] //~ ERROR: attribute should be applied to struct
enum ESimd { A, B }

#[repr(i8)]
enum EInt { A, B }

fn main() {}
