// compile-flags: --test
// run-pass

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
