// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-pass

#![feature(decl_macro)]

macro_rules! define_field {
    () => {
        struct S { field: u8 }
    }
}

macro use_field($define_field: item) {
    $define_field

    // OK, both struct name `S` and field `name` resolve to definitions produced by `define_field`
    // and living in the "root" context that is in scope at `use_field`'s def-site.
    fn f() { S { field: 0 }; }
}

use_field!(define_field!{});

fn main() {}
