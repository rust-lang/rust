#![expect(incomplete_features)]
#![feature(min_generic_const_args)]

trait GoodTr {
    type const NUM: usize;
}

struct BadS;

impl GoodTr for BadS {
    const NUM: usize = 42;
    //~^ ERROR implementation of a `type const` must also be marked as `type const`
}

fn accept_good_tr<const N: usize, T: GoodTr<NUM = { N }>>(_x: &T) {}

fn main() {
    accept_good_tr(&BadS);
}
