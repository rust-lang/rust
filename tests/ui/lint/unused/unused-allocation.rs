#![feature(box_syntax)]
#![deny(unused_allocation)]

fn main() {
    _ = (box [1]).len(); //~ error: unnecessary allocation, use `&` instead
    _ = Box::new([1]).len(); //~ error: unnecessary allocation, use `&` instead
}
