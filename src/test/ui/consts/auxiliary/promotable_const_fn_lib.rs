// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Crate that exports a const fn. Used for testing cross-crate.

#![feature(staged_api, rustc_attrs)]
#![stable(since="1.0.0", feature = "mep")]

#![crate_type="rlib"]

#[rustc_promotable]
#[stable(since="1.0.0", feature = "mep")]
#[inline]
pub const fn foo() -> usize { 22 }

#[stable(since="1.0.0", feature = "mep")]
pub struct Foo(usize);

impl Foo {
    #[stable(since="1.0.0", feature = "mep")]
    #[inline]
    #[rustc_promotable]
    pub const fn foo() -> usize { 22 }
}
