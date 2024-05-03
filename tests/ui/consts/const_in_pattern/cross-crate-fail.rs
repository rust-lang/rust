//@ aux-build:consts.rs

extern crate consts;

struct Defaulted;
impl consts::AssocConst for Defaulted {}

fn main() {
    let _ = Defaulted;
    match None {
        consts::SOME => panic!(),
        //~^ must be annotated with `#[derive(PartialEq)]`

        _ => {}
    }

    match None {
        <Defaulted as consts::AssocConst>::SOME  => panic!(),
        //~^ must be annotated with `#[derive(PartialEq)]`

        _ => {}
    }
}
