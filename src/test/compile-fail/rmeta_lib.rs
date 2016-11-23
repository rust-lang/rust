// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:rmeta_meta.rs
// no-prefer-dynamic
// error-pattern: crate `rmeta_meta` required to be available in rlib, but it was not available

// Check that building a non-metadata crate fails if a dependent crate is
// metadata-only.

extern crate rmeta_meta;
use rmeta_meta::Foo;

fn main() {
    let _ = Foo { field: 42 };
}
