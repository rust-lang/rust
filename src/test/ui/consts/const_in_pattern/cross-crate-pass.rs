// run-pass
// aux-build:consts.rs

#![warn(indirect_structural_match)]

extern crate consts;
use consts::CustomEq;

fn main() {
    match Some(CustomEq) {
        consts::NONE => panic!(),
        _ => {}
    }
}
