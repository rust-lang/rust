// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags:-Zprint-mono-items=eager
// ignore-tidy-linelength

#![feature(start)]

pub static FN : fn() = foo::<i32>;

pub fn foo<T>() { }

//~ MONO_ITEM fn static_init::foo[0]<i32>
//~ MONO_ITEM static static_init::FN[0]

//~ MONO_ITEM fn static_init::start[0]
#[start]
fn start(_: isize, _: *const *const u8) -> isize {
    0
}
