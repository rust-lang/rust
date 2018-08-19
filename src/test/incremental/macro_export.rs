// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// revisions: cfail1 cfail2 cfail3
// compile-pass

// This test case makes sure that we can compile with incremental compilation
// enabled when there are macros exported from this crate. (See #37756)

#![crate_type="rlib"]

#[macro_export]
macro_rules! some_macro {
    ($e:expr) => ($e + 1)
}
