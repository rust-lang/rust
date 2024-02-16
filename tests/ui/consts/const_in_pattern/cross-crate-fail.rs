//@ aux-build:consts.rs

#![warn(indirect_structural_match)]

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
