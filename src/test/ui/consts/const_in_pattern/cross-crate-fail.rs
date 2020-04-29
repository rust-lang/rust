// aux-build:consts.rs

#![warn(indirect_structural_match)]

extern crate consts;

fn main() {
    match None {
        consts::SOME => panic!(),
        //~^ must be annotated with `#[derive(PartialEq, Eq)]`
        //~| must be annotated with `#[derive(PartialEq, Eq)]`

        _ => {}
    }
}
