// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// gate-test-dropck_eyepatch

// Check that `may_dangle` is rejected if `dropck_eyepatch` feature gate is absent.

#![feature(generic_param_attrs)]

struct Pt<A>(A);
impl<#[may_dangle] A> Drop for Pt<A> {
    //~^ ERROR may_dangle has unstable semantics and may be removed in the future
    //~| HELP add #![feature(dropck_eyepatch)] to the crate attributes to enable
    fn drop(&mut self) { }
}
