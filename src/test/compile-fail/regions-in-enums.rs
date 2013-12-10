// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that lifetimes must be declared for use on enums.
// See also regions-undeclared.rs

enum yes0<'lt> {
    X3(&'lt uint)
}

enum yes1<'a> {
    X4(&'a uint)
}

enum no0 {
    X5(&'foo uint) //~ ERROR use of undeclared lifetime name `'foo`
}

enum no1 {
    X6(&'a uint) //~ ERROR use of undeclared lifetime name `'a`
}

fn main() {}
