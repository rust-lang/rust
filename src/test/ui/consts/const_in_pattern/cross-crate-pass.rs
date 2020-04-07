// run-pass
// aux-build:consts.rs

#![warn(indirect_structural_match)]

extern crate consts;
use consts::*;

fn main() {
    match Some(CustomEq) {
        NONE => panic!(),
        _ => {}
    }
}
