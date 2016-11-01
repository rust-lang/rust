// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This checks that RUST_TEST_THREADS not being 1, 2, ... is detected
// properly.

// error-pattern:should be a positive integer
// compile-flags: --test
// exec-env:RUST_TEST_THREADS=foo
// ignore-emscripten

#[test]
fn do_nothing() {}
