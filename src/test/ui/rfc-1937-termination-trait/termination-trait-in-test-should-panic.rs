// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: --test

#![feature(termination_trait_test)]
#![feature(test)]

extern crate test;
use std::num::ParseIntError;
use test::Bencher;

#[test]
#[should_panic]
fn not_a_num() -> Result<(), ParseIntError> {
    //~^ ERROR functions using `#[should_panic]` must return `()`
    let _: u32 = "abc".parse()?;
    Ok(())
}
