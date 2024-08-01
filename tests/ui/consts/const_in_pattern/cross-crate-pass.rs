//@ run-pass
//@ aux-build:consts.rs

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
