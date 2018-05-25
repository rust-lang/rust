// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-pass

#![deny(single_use_lifetimes)]
#![allow(dead_code)]
#![allow(unused_variables)]

// Test that we DO NOT warn when lifetime name is used only
// once in a fn return type -- using `'_` is not legal there,
// as it must refer back to an argument.
//
// (Normally, using `'static` would be preferred, but there are
// times when that is not what you want.)
//
// run-pass

fn b<'a>() -> &'a u32 { // OK: used only in return type
    &22
}

fn main() { }
