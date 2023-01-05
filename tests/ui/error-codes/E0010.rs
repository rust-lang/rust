#![feature(box_syntax)]
#![allow(warnings)]

const CON : Box<i32> = box 0; //~ ERROR E0010

fn main() {}
