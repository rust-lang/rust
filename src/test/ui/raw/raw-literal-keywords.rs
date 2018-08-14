// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -Z parse-only

#![feature(raw_identifiers)]

fn test_if() {
    r#if true { } //~ ERROR found `true`
}

fn test_struct() {
    r#struct Test; //~ ERROR found `Test`
}

fn test_union() {
    r#union Test; //~ ERROR found `Test`
}
