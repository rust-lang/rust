#![feature(box_syntax)]

fn main() {
    let f;
    f = box f;
    //~^ ERROR mismatched types
    //~| cyclic type of infinite size
}
