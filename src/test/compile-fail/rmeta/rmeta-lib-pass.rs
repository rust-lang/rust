// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: --emit=metadata
// aux-build:rmeta_rlib.rs
// no-prefer-dynamic
// must-compile-successfully

// Check that building a metadata crate works with a dependent, rlib crate.
// This is a cfail test since there is no executable to run.

extern crate rmeta_rlib;
use rmeta_rlib::Foo;

pub fn main() {
    let _ = Foo { field: 42 };
}
