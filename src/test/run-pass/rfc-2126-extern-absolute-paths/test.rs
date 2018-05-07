// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Check that `#[test]` works with extern-absolute-paths enabled.
//
// Regression test for #47075.

// compile-flags: --test --edition=2018 -Zunstable-options

#![feature(extern_absolute_paths)]

#[test]
fn test() {
}
