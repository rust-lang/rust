// run-pass
// pretty-expanded FIXME #23616

#![allow(path_statements)]
#![feature(box_syntax)]

pub fn main() {
    let y: Box<_> = box 1;
    y;
}
