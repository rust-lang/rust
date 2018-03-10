// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Forbidding a group (here, `unused`) overrules subsequent allowance of both
// the group, and an individual lint in the group (here, `unused_variables`);
// and, forbidding an individual lint (here, `non_snake_case`) overrules
// subsequent allowance of a lint group containing it (here, `bad_style`). See
// Issue #42873.

#![forbid(unused, non_snake_case)]

#[allow(unused_variables)] //~ ERROR overruled
fn foo() {}

#[allow(unused)] //~ ERROR overruled
fn bar() {}

#[allow(bad_style)] //~ ERROR overruled
fn main() {
    println!("hello forbidden world")
}
