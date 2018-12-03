// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This test is just checking that we won't ICE if logging is turned
// on; don't bother trying to compare that (copious) output. (Note
// also that this test potentially silly, since we do not build+test
// debug versions of rustc as part of our continuous integration
// process...)
//
// dont-check-compiler-stdout
// dont-check-compiler-stderr
// compile-flags: --error-format human

// rustc-env:RUST_LOG=debug

fn main() {}
