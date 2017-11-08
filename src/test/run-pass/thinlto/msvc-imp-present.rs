// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:msvc-imp-present.rs
// compile-flags: -Z thinlto -C codegen-units=8
// min-llvm-version: 4.0
// no-prefer-dynamic

// On MSVC we have a "hack" where we emit symbols that look like `_imp_$name`
// for all exported statics. This is done because we apply `dllimport` to all
// imported constants and this allows everything to actually link correctly.
//
// The ThinLTO passes aggressively remove symbols if they can, and this test
// asserts that the ThinLTO passes don't remove these compiler-generated
// `_imp_*` symbols. The external library that we link in here is compiled with
// ThinLTO and multiple codegen units and has a few exported constants. Note
// that we also namely compile the library as both a dylib and an rlib, but we
// link the rlib to ensure that we assert those generated symbols exist.

extern crate msvc_imp_present as bar;

fn main() {
    println!("{}", bar::A);
}
