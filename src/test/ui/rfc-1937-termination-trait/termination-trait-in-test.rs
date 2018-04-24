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
// run-pass

#![feature(termination_trait_test)]
#![feature(test)]

extern crate test;
use std::num::ParseIntError;
use test::Bencher;

#[test]
fn is_a_num() -> Result<(), ParseIntError> {
    let _: u32 = "22".parse()?;
    Ok(())
}

#[bench]
fn test_a_positive_bench(_: &mut Bencher) -> Result<(), ParseIntError> {
    Ok(())
}

#[bench]
#[should_panic]
fn test_a_neg_bench(_: &mut Bencher) -> Result<(), ParseIntError> {
    let _: u32 = "abc".parse()?;
    Ok(())
}
