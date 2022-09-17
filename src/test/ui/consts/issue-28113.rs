#![allow(warnings)]

const X: u8 =
    || -> u8 { 5 }()
    //~^ ERROR the trait bound
;

fn main() {}
