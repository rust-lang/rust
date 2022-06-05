#![feature(box_syntax)]
#![deny(unused_allocation)]

fn main() {
    let foo = (box [1, 2, 3]).len(); //~ ERROR: unnecessary allocation
    let one = (box vec![1]).pop(); //~ ERROR: unnecessary allocation

    let foo = Box::new([1, 2, 3]).len(); //~ ERROR: unnecessary allocation
    let one = Box::new(vec![1]).pop(); //~ ERROR: unnecessary allocation
}
