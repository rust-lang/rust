#![expect(incomplete_features)]
#![feature(min_generic_const_args)]

trait GoodTr {
    #[type_const]
    const NUM: usize;
}

struct BadS;

impl GoodTr for BadS {
    const NUM: usize = 42;
    //~^ ERROR implementation of `#[type_const]` const must be marked with `#[type_const]`
}

fn accept_good_tr<const N: usize, T: GoodTr<NUM = { N }>>(_x: &T) {}

fn main() {
    accept_good_tr(&BadS);
}
