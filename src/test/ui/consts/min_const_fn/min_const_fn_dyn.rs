// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(min_const_fn)]

struct HasDyn {
    field: &'static dyn std::fmt::Debug,
}

struct Hide(HasDyn);

const fn no_inner_dyn_trait(_x: Hide) {}
const fn no_inner_dyn_trait2(x: Hide) {
    x.0.field;
//~^ ERROR trait bounds other than `Sized`
}
const fn no_inner_dyn_trait_ret() -> Hide { Hide(HasDyn { field: &0 }) }
//~^ ERROR trait bounds other than `Sized`

fn main() {}
