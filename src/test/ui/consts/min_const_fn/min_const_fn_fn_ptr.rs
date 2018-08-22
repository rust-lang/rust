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

struct HasPtr {
    field: fn(),
}

struct Hide(HasPtr);

fn field() {}

const fn no_inner_dyn_trait(_x: Hide) {}
const fn no_inner_dyn_trait2(x: Hide) {
    x.0.field;
//~^ ERROR function pointers in const fn
}
const fn no_inner_dyn_trait_ret() -> Hide { Hide(HasPtr { field }) }
//~^ ERROR function pointers in const fn

fn main() {}
