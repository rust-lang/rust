// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Regression test for a problem with the first mod attribute
// being applied to every mod

#[cfg(target_os = "linux")]
mod hello;

#[cfg(target_os = "macos")]
mod hello;

#[cfg(target_os = "win32")]
mod hello;

#[cfg(target_os = "freebsd")]
mod hello;

#[cfg(target_os = "android")]
mod hello;

pub fn main() { }
