// run-pass
// aux-build:consts.rs

#![warn(indirect_structural_match)]

extern crate consts;
use consts::CustomEq;

struct Defaulted;
impl consts::AssocConst for Defaulted {}

fn main() {
    let _ = Defaulted;
    match Some(CustomEq) {
        consts::NONE => panic!(),
        _ => {}
    }

    match Some(CustomEq) {
        <Defaulted as consts::AssocConst>::NONE  => panic!(),
        _ => {}
    }
}
