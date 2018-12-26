// run-pass
#![allow(dead_code)]
// pretty-expanded FIXME #23616

#![feature(box_syntax)]

pub fn main() {
    let _: Box<_> = box 100;
}

fn vec() {
    vec![0];
}
