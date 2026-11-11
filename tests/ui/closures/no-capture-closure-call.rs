//! Sanity check for no capture closures

//@ run-pass

pub fn main() {
    let _x: Box<_> = Box::new(1);
    let lam_move = || {};
    lam_move();
}
