// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:rustdoc-ffi.rs
// ignore-android

extern crate rustdoc_ffi as lib;

// @has ffi/fn.foreigner.html //pre 'pub unsafe extern fn foreigner(cold_as_ice: u32)'
pub use lib::foreigner;

extern "C" {
    // @has ffi/fn.another.html //pre 'pub unsafe extern fn another(cold_as_ice: u32)'
    pub fn another(cold_as_ice: u32);
}
