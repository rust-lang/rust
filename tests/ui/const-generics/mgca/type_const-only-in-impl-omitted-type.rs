#![expect(incomplete_features)]
#![feature(min_generic_const_args)]

trait BadTr {
    const NUM: usize;
}

struct GoodS;

impl BadTr for GoodS {
    type const NUM: = 84;
    //~^ ERROR: missing type for `const` item
    //~| ERROR: type annotations needed for the literal

}

fn accept_bad_tr<const N: usize, T: BadTr<NUM = { N }>>(_x: &T) {}
//~^ ERROR use of trait associated const not defined as `type const`

fn main() {
    accept_bad_tr::<84, _>(&GoodS);
}
