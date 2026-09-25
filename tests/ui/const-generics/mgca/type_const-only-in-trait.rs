#![expect(incomplete_features)]
#![feature(gca_min_const_items)]

use std::gca;

trait GoodTr {
    #[rustc_always_gca]
    const NUM: usize;
}

struct BadS;

impl GoodTr for BadS {
    const NUM: usize = 42;
    //~^ ERROR implementation of a `#[rustc_always_gca]` must have a `gca!` RHS
}

fn accept_good_tr<const N: usize, T: GoodTr<NUM = { N }>>(_x: &T) {}

fn main() {
    accept_good_tr(&BadS);
}
