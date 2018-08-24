#![feature(box_syntax)]

fn main() {
    let f;
    let g;
    g = f;
    f = box g;
    //~^  ERROR mismatched types
    //~| cyclic type of infinite size
}
