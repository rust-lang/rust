// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This test checks variations on `<#[attr] 'a, #[oops]>`, where
// `#[oops]` is left dangling (that is, it is unattached, with no
// formal binding following it).

#![feature(generic_param_attrs, rustc_attrs)]
#![allow(dead_code)]

struct RefAny<'a, T>(&'a T);

impl<#[rustc_1] 'a, #[rustc_2] T, #[oops]> RefAny<'a, T> {
    //~^ ERROR expected identifier, found `>`
}

fn main() {

}
