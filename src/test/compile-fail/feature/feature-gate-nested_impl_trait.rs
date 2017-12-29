// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
#![feature(conservative_impl_trait, universal_impl_trait)]

use std::fmt::Debug;

fn fine(x: impl Into<u32>) -> impl Into<u32> { x }

fn bad_in_ret_position(x: impl Into<u32>) -> impl Into<impl Debug> { x }
//~^ ERROR nested `impl Trait` is experimental

fn bad_in_fn_syntax(x: fn() -> impl Into<impl Debug>) {}
//~^ ERROR nested `impl Trait` is experimental

fn bad_in_arg_position(_: impl Into<impl Debug>) { }
//~^ ERROR nested `impl Trait` is experimental

struct X;
impl X {
    fn bad(x: impl Into<u32>) -> impl Into<impl Debug> { x }
    //~^ ERROR nested `impl Trait` is experimental
}

fn allowed_in_assoc_type() -> impl Iterator<Item=impl Fn()> {
    vec![|| println!("woot")].into_iter()
}

fn allowed_in_ret_type() -> impl Fn() -> impl Into<u32> {
    || 5
}

fn main() {}
