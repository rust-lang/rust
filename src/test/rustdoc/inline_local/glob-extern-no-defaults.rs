// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: --no-defaults

#![crate_name = "foo"]

mod mod1 {
    extern {
        pub fn public_fn();
        fn private_fn();
    }
}

pub use mod1::*;

// @has foo/index.html
// @has - "mod1"
// @has - "public_fn"
// @!has - "private_fn"
// @has foo/fn.public_fn.html
// @!has foo/fn.private_fn.html

// @has foo/mod1/index.html
// @has - "public_fn"
// @has - "private_fn"
// @has foo/mod1/fn.public_fn.html
// @has foo/mod1/fn.private_fn.html
