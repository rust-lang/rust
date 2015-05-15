// Copyright 2013-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:macro_crate_test.rs
// ignore-cross-compile
//
// macro_crate_test will not compile on a cross-compiled target because
// libsyntax is not compiled for it.

#![deny(plugin_as_library)]

extern crate macro_crate_test; //~ ERROR compiler plugin used as an ordinary library

fn main() { }
