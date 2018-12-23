// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![unstable(feature = "humans",
            reason = "who ever let humans program computers,
            we're apparently really bad at it",
            issue = "0")]

#![feature(rustc_const_unstable, const_fn, foo, foo2)]
#![feature(staged_api)]

#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_unstable(feature="foo")]
const fn foo() -> u32 { 42 }

#[stable(feature = "rust1", since = "1.0.0")]
// can't call non-min_const_fn
const unsafe fn bar() -> u32 { foo() } //~ ERROR can only call other `min_const_fn`

#[unstable(feature = "rust1", issue="0")]
const fn foo2() -> u32 { 42 }

#[stable(feature = "rust1", since = "1.0.0")]
// can't call non-min_const_fn
const unsafe fn bar2() -> u32 { foo2() } //~ ERROR can only call other `min_const_fn`

// check whether this function cannot be called even with the feature gate active
#[unstable(feature = "foo2", issue="0")]
const fn foo2_gated() -> u32 { 42 }

#[stable(feature = "rust1", since = "1.0.0")]
// can't call non-min_const_fn
const unsafe fn bar2_gated() -> u32 { foo2_gated() } //~ ERROR can only call other `min_const_fn`

fn main() {}
