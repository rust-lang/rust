#![feature(box_syntax)]
// This disables the test completely:
// ignore-stage1

fn main() {
    // With the nested Vec, this is calling Offset(Unique::empty(), 0).
    let args : Vec<Vec<i32>> = Vec::new();
    let local = box args;
}
