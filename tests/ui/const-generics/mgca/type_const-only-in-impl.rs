//@ check-pass

#![expect(incomplete_features)]
#![feature(min_generic_const_args)]

trait BadTr {
    const NUM: usize;
}

struct GoodS;

impl BadTr for GoodS {
    type const NUM: usize = 84;
}

fn accept_bad_tr<const N: usize, T: BadTr<NUM = { N }>>(_x: &T) {}

fn main() {
    accept_bad_tr::<84, _>(&GoodS);
}
