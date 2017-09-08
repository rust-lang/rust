// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that DynSized cannot be implemented manually.

#![feature(extern_types)]
#![feature(dynsized)]

use std::marker::DynSized;

extern {
    type foo;
}

impl DynSized for foo { }
//~^ ERROR explicit impls for the `DynSized` trait are not permitted

fn main() { }
