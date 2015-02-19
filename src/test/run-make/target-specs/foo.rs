// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(lang_items, no_std)]
#![no_std]

#[lang="phantom_fn"]
trait PhantomFn<A:?Sized,R:?Sized=()> { }
impl<A:?Sized, R:?Sized, U:?Sized> PhantomFn<A,R> for U { }

#[lang="copy"]
trait Copy : PhantomFn<Self> { }

#[lang="sized"]
trait Sized : PhantomFn<Self>  { }

#[lang="start"]
fn start(_main: *const u8, _argc: int, _argv: *const *const u8) -> int { 0 }

extern {
    fn _foo() -> [u8; 16];
}

fn _main() {
    let _a = unsafe { _foo() };
}
