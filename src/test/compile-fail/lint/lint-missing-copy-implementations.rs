// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// See issue 19712

#![deny(missing_copy_implementations)]

mod inner {
    pub struct Foo { //~ ERROR type could implement `Copy`; consider adding `impl Copy`
        pub field: i32
    }
}

pub fn foo() -> inner::Foo {
    inner::Foo { field: 42 }
}

fn main() {}
