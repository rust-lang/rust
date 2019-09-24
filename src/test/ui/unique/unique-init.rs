// run-pass
// pretty-expanded FIXME #23616

#![feature(box_syntax)]

pub fn main() {
    let _i: Box<_> = box 100;
}
