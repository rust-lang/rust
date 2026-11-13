//@ check-pass
#![allow(incomplete_features)]
#![feature(mut_ref)]
fn main() {
    let mut ref x = 10;
    x = &11;
    let ref mut y = 12;
    *y = 13;
    let mut ref mut z = 14;
    z = &mut 15;
}
