#![feature(box_syntax)]

use std::cell::Cell;

pub fn main() {
    let x: Box<_> = box Cell::new(5);
    x.set(1000);
    println!("{}", x.get());
}
