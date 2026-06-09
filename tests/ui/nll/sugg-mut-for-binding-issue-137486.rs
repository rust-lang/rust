//@ run-rustfix
#![allow(unused_assignments)]

use std::pin::Pin;
fn main() {
    let mut s = String::from("hello");
    let mut ref_s = &mut s;

    ref_s = &mut String::from("world"); //~ ERROR temporary value dropped while borrowed [E0716]

    print!("r1 = {}", ref_s);

    let mut val: u8 = 5;
    let mut s = Pin::new(&mut val);
    let mut ref_s = &mut s;

    let mut val2: u8 = 10;
    ref_s = &mut Pin::new(&mut val2); //~ ERROR temporary value dropped while borrowed [E0716]

    print!("r1 = {}", ref_s);
}
