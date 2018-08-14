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
// aux-build:rmeta_meta.rs
// no-prefer-dynamic

// Check that building a metadata crate finds an error with a dependent,
// metadata-only crate.


extern crate rmeta_meta;
use rmeta_meta::Foo;

fn main() {
    let _ = Foo { field2: 42 }; //~ ERROR struct `rmeta_meta::Foo` has no field named `field2`
}
