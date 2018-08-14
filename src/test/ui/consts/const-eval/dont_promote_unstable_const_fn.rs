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

#![feature(rustc_const_unstable, const_fn)]
#![feature(staged_api)]

#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_unstable(feature="foo")]
const fn foo() -> u32 { 42 }

fn meh() -> u32 { 42 }

const fn bar() -> u32 { foo() } //~ ERROR `foo` is not yet stable as a const fn

fn a() {
    let _: &'static u32 = &foo(); //~ ERROR does not live long enough
}

fn main() {
    let _: &'static u32 = &meh(); //~ ERROR does not live long enough
    let x: &'static _ = &std::time::Duration::from_millis(42).subsec_millis();
}
