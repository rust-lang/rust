#![feature(box_syntax)]
#![allow(unused_variables)]
#![deny(unreachable_code)]

fn main() {
    let x = box return; //~ ERROR unreachable
    println!("hi");
}
