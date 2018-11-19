#![feature(test, rustc_private)]

extern crate test;
use crate::test::Bencher;
mod helpers;
use crate::helpers::*;

#[bench]
fn repeat(bencher: &mut Bencher) {
    miri_helper::run("repeat", bencher);
}

#[bench]
fn repeat_manual(bencher: &mut Bencher) {
    miri_helper::run("repeat_manual", bencher);
}
