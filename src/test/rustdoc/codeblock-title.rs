// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![crate_name = "foo"]

// ignore-tidy-linelength

// @has foo/fn.bar.html '//*[@class="tooltip compile_fail"]/span' "This example deliberately fails to compile"
// @has foo/fn.bar.html '//*[@class="tooltip ignore"]/span' "This example is not tested"

/// foo
///
/// ```compile_fail
/// foo();
/// ```
///
/// ```ignore (tidy)
/// goo();
/// ```
///
/// ```
/// let x = 0;
/// ```
pub fn bar() -> usize { 2 }
