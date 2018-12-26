// pretty-expanded FIXME #23616

#![feature(box_syntax)]

pub fn main() {
    let _x: Box<_> = box 1;
    let lam_move = || {};
    lam_move();
}
