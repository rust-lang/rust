#![allow(warnings)]

const X: u8 =
    || -> u8 { 5 }()
    //~^ ERROR cannot call non-const fn
;

fn main() {}
