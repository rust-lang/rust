#![feature(inherent_associated_types)]
#![allow(incomplete_features)]

struct Parameterized<T, U>(T, U);

impl Parameterized<(), ()> {
    type Output = bool;
}

impl<T> Parameterized<bool, T> {
    type Result = T;
}

fn main() {
    let _: Parameterized<(), ()>::Output = String::new(); //~ ERROR mismatched types
    let _: Parameterized<bool, u32>::Result = (); //~ ERROR mismatched types
}
