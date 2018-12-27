// pretty-expanded FIXME #23616

#![feature(box_syntax)]

pub fn main() {
    let x: Box<isize> = box 10;
    let _y: isize = *x;
}
