// run-pass
// pretty-expanded FIXME #23616

#![feature(box_syntax)]

pub fn main() {
    let _x: Box<_> = box vec![0,0,0,0,0];
}
