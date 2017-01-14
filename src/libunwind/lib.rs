// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![no_std]
#![crate_name = "unwind"]
#![crate_type = "rlib"]
#![unstable(feature = "panic_unwind", issue = "32837")]
#![deny(warnings)]

#![feature(cfg_target_vendor)]
#![feature(staged_api)]
#![feature(unwind_attributes)]

#![cfg_attr(not(target_env = "msvc"), feature(libc))]

#[cfg(not(target_env = "msvc"))]
extern crate libc;

#[cfg(not(target_env = "msvc"))]
mod libunwind;
#[cfg(not(target_env = "msvc"))]
pub use libunwind::*;
