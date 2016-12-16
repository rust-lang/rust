// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
#![allow(non_camel_case_types)]
#![cfg_attr(not(stage0), feature(i128_type))]
#![no_std]
#![crate_type="rlib"]
#![crate_name="rustc_i128"]

#[cfg(stage0)]
pub type i128 = i64;
#[cfg(stage0)]
pub type u128 = u64;

#[cfg(not(stage0))]
pub type i128 = int::_i128;
#[cfg(not(stage0))]
pub type u128 = int::_u128;
#[cfg(not(stage0))]
mod int {
    pub type _i128 = i128;
    pub type _u128 = u128;
}
