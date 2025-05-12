//@ aux-build:consts.rs

extern crate consts;

struct Defaulted;
impl consts::AssocConst for Defaulted {}

fn main() {
    let _ = Defaulted;
    match None {
        consts::SOME => panic!(),
        //~^ ERROR constant of non-structural type `CustomEq` in a pattern
        _ => {}
    }

    match None {
        <Defaulted as consts::AssocConst>::SOME  => panic!(),
        //~^ ERROR constant of non-structural type `CustomEq` in a pattern
        _ => {}
    }
}
