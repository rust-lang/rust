// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(no_std)]
#![no_std]
#![feature(lang_items)]

#[lang="phantom_fn"]
pub trait PhantomFn<A:?Sized,R:?Sized=()> { }
impl<A:?Sized, R:?Sized, U:?Sized> PhantomFn<A,R> for U { }

#[lang="sized"]
pub trait Sized : PhantomFn<Self> {}

#[lang="panic"]
fn panic(_: &(&'static str, &'static str, uint)) -> ! { loop {} }

#[lang = "stack_exhausted"]
extern fn stack_exhausted() {}

#[lang = "eh_personality"]
extern fn eh_personality() {}

#[lang="copy"]
pub trait Copy : PhantomFn<Self> {
    // Empty.
}


