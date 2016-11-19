#![feature(test, rustc_private)]

extern crate test;
use test::Bencher;
mod helpers;
use helpers::*;

#[bench]
fn repeat(bencher: &mut Bencher) {
    miri_helper::run("repeat", bencher);
}

#[bench]
fn repeat_manual(bencher: &mut Bencher) {
    miri_helper::run("repeat_manual", bencher);
}
