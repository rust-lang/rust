//@ known-bug: #120793
#![feature(effects)]

trait Dim {
    fn dim() -> usize;
}

enum Dim3 {}

impl Dim for Dim3 {
    fn dim(mut x: impl Iterator<Item = &'_ ()>) -> usize {
        3
    }
}

fn main() {
    let array: [usize; Dim3::dim()]
    //~^ ERROR E0015
        = [0; Dim3::dim()];
        //~^ ERROR E0015
}
