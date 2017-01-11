// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![no_implicit_prelude]

// Test that things from the prelude aren't in scope. Use many of them
// so that renaming some things won't magically make this test fail
// for the wrong reason (e.g. if `Add` changes to `Addition`, and
// `no_implicit_prelude` stops working, then the `impl Add` will still
// fail with the same error message).

struct Test;
impl Add for Test {} //~ ERROR cannot find trait `Add` in this scope
impl Clone for Test {} //~ ERROR cannot find trait `Clone` in this scope
impl Iterator for Test {} //~ ERROR cannot find trait `Iterator` in this scope
impl ToString for Test {} //~ ERROR cannot find trait `ToString` in this scope
impl Writer for Test {} //~ ERROR cannot find trait `Writer` in this scope

fn main() {
    drop(2) //~ ERROR cannot find function `drop` in this scope
}
