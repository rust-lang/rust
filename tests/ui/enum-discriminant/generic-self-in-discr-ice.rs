#![feature(sized_hierarchy)]

use std::marker::PointeeSized;

#[repr(usize)]
enum What<T: PointeeSized> {
    X = size_of::<*mut Self>(),
    //~^ ERROR generic parameters may not be used in enum discriminant values
    Y(*mut T),
}

fn main() {}
