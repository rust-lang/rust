// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(const_transmute)]

use std::mem;

const UNALIGNED: &u16 = unsafe { mem::transmute(&[0u8; 4]) };
//~^ ERROR this constant likely exhibits undefined behavior

const NULL: &u16 = unsafe { mem::transmute(0usize) };
//~^ ERROR this constant likely exhibits undefined behavior

const REF_AS_USIZE: usize = unsafe { mem::transmute(&0) };
//~^ ERROR this constant likely exhibits undefined behavior

const REF_AS_USIZE_SLICE: &[usize] = &[unsafe { mem::transmute(&0) }];
//~^ ERROR this constant likely exhibits undefined behavior

const USIZE_AS_REF: &'static u8 = unsafe { mem::transmute(1337usize) };
//~^ ERROR this constant likely exhibits undefined behavior

fn main() {}
