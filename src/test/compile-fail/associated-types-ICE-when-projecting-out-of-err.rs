// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that we do not ICE when the self type is `ty::err`, but rather
// just propagate the error.

#![crate_type = "lib"]
#![feature(lang_items)]
#![feature(no_core)]
#![no_core]

#[lang="sized"]
pub trait Sized {
    // Empty.
}

#[lang = "add"]
trait Add<RHS=Self> {
    type Output;

    fn add(self, RHS) -> Self::Output;
}

fn ice<A>(a: A) {
    let r = loop {};
    r = r + a;
    //~^ ERROR the trait bound `(): Add<A>` is not satisfied
    //~| NOTE the trait `Add<A>` is not implemented for `()`
}
