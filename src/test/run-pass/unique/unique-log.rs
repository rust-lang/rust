// run-pass
#![feature(box_syntax)]

pub fn main() {
    let i: Box<_> = box 100;
    println!("{}", i);
}
