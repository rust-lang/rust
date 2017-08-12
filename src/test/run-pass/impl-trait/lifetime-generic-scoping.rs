// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(conservative_impl_trait)]

use std::fmt::Debug;

fn foo<'a>(x: &'a u32) -> impl Debug { *x }
fn foo_elided(x: &u32) -> impl Debug { *x }

fn main() {
    // Make sure that the lifetime parameter of `foo` isn't included in `foo`'s return type:
    let _ = {
        let x = 5;
        foo(&x)
    };
    let _ = {
        let x = 5;
        foo_elided(&x)
    };
}
