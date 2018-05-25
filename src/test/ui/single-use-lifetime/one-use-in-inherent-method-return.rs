// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![deny(single_use_lifetimes)]
#![allow(dead_code)]
#![allow(unused_variables)]

// Test that we DO NOT warn for a lifetime used just once in a return type,
// where that return type is in an inherent method.

struct Foo<'f> {
    data: &'f u32
}

impl<'f> Foo<'f> { //~ ERROR `'f` only used once
    fn inherent_a<'a>(&self) -> &'a u32 { // OK for 'a
        &22
    }
}

fn main() { }
