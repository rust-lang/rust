// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This is testing whether `#[inline]` signals an error or warning
// when put in "weird" places.
//
// (This file sits on its own because it actually signals an error,
// which would mess up the treatment of other cases in
// issue-43106-gating-of-builtin-attrs.rs)

// Crate-level is accepted, though it is almost certainly unused?
#![inline                     = "2100"]

#[inline = "2100"]
//~^ ERROR attribute should be applied to function or closure
mod inline {
    mod inner { #![inline="2100"] }
    //~^ ERROR attribute should be applied to function or closure

    #[inline = "2100"] fn f() { }

    #[inline = "2100"] struct S;
    //~^ ERROR attribute should be applied to function or closure

    #[inline = "2100"] type T = S;
    //~^ ERROR attribute should be applied to function or closure

    #[inline = "2100"] impl S { }
    //~^ ERROR attribute should be applied to function or closure
}

fn main() {}
