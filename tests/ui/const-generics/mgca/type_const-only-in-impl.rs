#![expect(incomplete_features)]
#![feature(gca_min_const_items)]

use std::gca;

trait BadTr {
    const NUM: usize;
}

struct GoodS;

impl BadTr for GoodS {
    const NUM: usize = gca!(84);
    //~^ ERROR implementation of a regular const cannot have a `gca!` RHS
}

fn accept_bad_tr<const N: usize, T: BadTr<NUM = { N }>>(_x: &T) {}
//~^ ERROR use of trait associated const not defined as `#[rustc_always_gca]`

fn main() {
    accept_bad_tr::<84, _>(&GoodS);
}
