//@ compile-flags: --test

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
