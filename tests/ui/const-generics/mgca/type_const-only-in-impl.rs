#![expect(incomplete_features)]
#![feature(min_generic_const_args)]

trait BadTr {
    const NUM: usize;
}

struct GoodS;

impl BadTr for GoodS {
    #[type_const]
    const NUM: usize = 84;
}

fn accept_bad_tr<const N: usize, T: BadTr<NUM = { N }>>(_x: &T) {}
//~^ ERROR use of trait associated const without `#[type_const]`

fn main() {
    accept_bad_tr::<84, _>(&GoodS);
}
